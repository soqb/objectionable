#![cfg_attr(
    any(feature = "strict-provenance", miri),
    feature(strict_provenance, ptr_metadata)
)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
//! Objectionable storage of `?Sized` types inline inside allocated objects.
//!
//! See [`BigBox`] for the crate's central API.
//!
//! # Soundness
//!
//! When the <span class="stab portability"><code>strict-provenance</code></span> crate feature is enabled,
//! this crate uses unstable APIs[^1] for sound execution under [the experimental strict provenance memory model]
//! and to support passing the test suite under [Miri].
//!
//! When this feature is disabled, this crate tries to replicate the unstable methods
//! using only those available in stable Rust.
//! This method is reliable enough for compilation in a practical environment,
//! but is considered unsound under the strict provenance memory model.
//! This should not have an effect on typical use cases (as of Rust 1.80),
//! but when interacting with strict provenance, *the aforementioned feature is required*.
//!
//! Bear in mind that this project is developped as a hobby, and is not formally verified.
//! There may be undiscovered unsoundness in untested edge cases,
//! issues and PRs are welcome at the linked repository page.
//!
//! [^1]: Notably, Rust's
//!     <a class="stab portability" href="https://github.com/rust-lang/rust/issues/95228"><code>strict_provenance</code></a>
//!     and
//!     <a class="stab portability" href="https://github.com/rust-lang/rust/issues/81513"><code>ptr_metadata</code></a>
//!     features are enabled.
//!
//! [the experimental strict provenance memory model]: core::ptr#strict-provenance
//! [Miri]: https://github.com/rust-lang/miri
//!
//! # Examples
//!
//! ```rust
//! // we define the trait object (and impls) which our `BigBox` will contain.
//! trait Character {
//!     fn personality(&self) -> &'static str;
//! }
//!
//! impl Character for u8 {
//!     fn personality(&self) -> &'static str {
//!         "very small"
//!     }
//! }
//!
//! impl Character for [u8; 1000] {
//!     fn personality(&self) -> &'static str {
//!         "enormous"
//!     }
//! }
//!
//! // implementing the unsafe trait `FromSized` is necessary for creating `BigBox` values,
//! // but there is thankfully a safe macro for this:
//! objectionable::impl_from_sized_for_trait_object!(dyn Character);
//!
//! // to use `BigBox`, we have to configure it with an internal type and maximum inline size:
//! type MyBox = objectionable::BigBox<dyn Character, 32>;
//!
//! // if we have a pre-allocated value, we can use `BigBox::new_boxed` to create a `BigBox` value:
//! let pre_boxed = Box::new(5_u8);
//! let pre_boxed = MyBox::new_boxed(pre_boxed);
//!
//! // alternatively, we can use `BigBox::new` which will store the value inline if it is small enough:
//! let inline = MyBox::new(2_u8);
//! assert!(inline.is_inline() == true);
//!
//! // but if the value is too large, it will be allocated on the heap via a standard `Box`:
//! let boxed = MyBox::new([2u8; 1000]);
//! assert!(boxed.is_inline() == false);
//!
//! // accessing values is easy:
//! assert_eq!(Character::personality(pre_boxed.as_ref()), "very small");
//! assert_eq!(Character::personality(inline.as_ref()), "very small");
//! assert_eq!(Character::personality(boxed.as_ref()), "enormous");
//! ```

use std::{
    alloc::Layout,
    cell::UnsafeCell,
    mem::{self, ManuallyDrop, MaybeUninit},
    ptr,
};

/// Similar to `Box<T>`, but stores small values inline.
///
/// `T` should be a `!Sized` type; otherwise, this type is redundant
/// because `Sized` types are either always small or always large.
///
/// [`BigBox`] values are either considered "boxed" (i.e. stored in a `Box`),
/// or "inline" (stored directly in the `BigBox` allocated object, without indirection).
///
/// # Which values can be stored inline?
///
/// **NB:** These details are subject to change,
/// but will always be accurately documented.
///
/// Types are considered small enough to store inline
/// if they have both of the following properties:
/// * A size no more than the `N` generic parameter.
/// * An alignment no more than than 8.
///   - Note, there is a generic parameter (`A`) intended for altering this limit,
///     but due to language restrictions, it cannot currently be used.
///
/// # Layout
///
/// The in-memory layout of this type should not be relied upon,
/// and should be considered unstable.
#[cfg_attr(not(doc), repr(C, align(8)))]
pub struct BigBox<T: ?Sized, const N: usize, const A: usize = 8> {
    inline: UnsafeCell<[MaybeUninit<u8>; N]>,
    // if the data ptr of `boxed` is non-null,
    // it is a pointer to the actual data,
    // constructed with `Box::into_raw`.
    // otherwise (if `boxed` is null),
    // the data is stored in `inline`.
    //
    // this is kinda freaky, but the opsem & layout of vtables
    // aren't at all guaranteed, so we have store the whole ptr.
    boxed: *mut T,
}

unsafe impl<T: Send + ?Sized, const N: usize, const A: usize> Send for BigBox<T, N, A> {}
unsafe impl<T: Sync + ?Sized, const N: usize, const A: usize> Sync for BigBox<T, N, A> {}

/// This is effectively `data_ptr.with_metadata_of(boxed)` on stable.
///
/// Note it also reuses the provenance of `meta_ptr`,
/// which is incorrect but there's nothing we can do on stable.
fn copy_metadata<T: ?Sized>(meta_ptr: *const T, data_ptr: *const u8) -> *const T {
    #[cfg(not(any(feature = "strict-provenance", miri)))]
    return meta_ptr
        .wrapping_byte_sub(meta_ptr as *const u8 as usize)
        .wrapping_byte_add(data_ptr as usize);
    #[cfg(any(feature = "strict-provenance", miri))]
    return ptr::from_raw_parts(data_ptr, ptr::metadata(meta_ptr));
}

fn take_into_box<T: ?Sized, const N: usize, const A: usize>(
    mut value: InlineBox<T, N, A>,
) -> Box<T> {
    // we have to implement `Box::new` manually because all this is all FREAKISH
    // and EVIL.

    let layout = Layout::for_value::<T>(value.as_ref());

    let alloc: *mut u8 = if layout.size() == 0 {
        // NB: we're not going through the "canonical"
        // way of constructing a ZST pointer (`NonNull::dangling`)
        // because we don't have static access to the value's layout.
        #[cfg(not(any(feature = "strict-provenance", miri)))]
        // SAFETY: GOD this is HORRIBLE i HATE EVERYTHING.
        unsafe {
            mem::transmute(layout.align())
        }
        #[cfg(any(feature = "strict-provenance", miri))]
        ptr::without_provenance_mut(layout.align())
    } else {
        let alloc = unsafe { std::alloc::alloc(layout) };
        if alloc.is_null() {
            std::alloc::handle_alloc_error(layout)
        }

        alloc
    };

    let ptr = ptr::from_mut::<T>(value.as_mut()) as *mut T;
    unsafe { alloc.copy_from_nonoverlapping(ptr.cast(), layout.size()) };

    let alloc = copy_metadata(ptr, alloc).cast_mut();
    unsafe { Box::from_raw(alloc) }
}

/// Allows construction of `?Sized` values from `Sized` values.
///
/// # Safety
///
/// The implementation of [`BigBox::new`] requires that
/// [`FromSized::fatten_pointer`] upholds the following conditions:
/// * The returned pointer must have the correct pointer metadata for the type.
/// * The returned pointer must be otherwise identical to the one passed.
///
/// For trait objects, this should be implemented through the trivial cast `thin as *mut Self`,
/// and this can be done safely with the [`impl_from_sized_for_trait_object` macro].
///
/// [`impl_from_sized_for_trait_object` macro]: impl_from_sized_for_trait_object
///
/// Note that there is no requirement that the "fat" pointer is even larger than the "thin" pointer,
/// only that the cast preserves the above semantics.
pub unsafe trait FromSized<T> {
    /// Extends a thin pointer with the metadata for its type.
    fn fatten_pointer(thin: *mut T) -> *mut Self;
}

/// Safely implements [`FromSized`] for trait objects.
///
/// ```rust
/// # use objectionable::impl_from_sized_for_trait_object;
/// trait MyTrait {}
///
/// // this invocation:
/// impl_from_sized_for_trait_object!(dyn MyTrait);
///
/// // expands to this implementation:
/// # #[cfg(any())]
/// impl<T: MyTrait> FromSized<T> for dyn MyTrait {}
/// ```
#[macro_export]
macro_rules! impl_from_sized_for_trait_object {
    (dyn $trait:path) => {
        // SAFETY: sized-type-to-trait-object pointer cast uses the vtable as the pointer metadata.
        unsafe impl<T: $trait> $crate::FromSized<T> for dyn $trait {
            fn fatten_pointer(thin: *mut T) -> *mut Self {
                thin as *mut Self
            }
        }
    };
}

// SAFETY: reflexive `fatten_pointer` returns the exact same pointer.
unsafe impl<T> FromSized<T> for T {
    fn fatten_pointer(thin: *mut T) -> *mut T {
        thin
    }
}

// SAFETY: array-to-slice pointer cast uses the array length as the pointer metadata.
unsafe impl<T, const N: usize> FromSized<[T; N]> for [T] {
    fn fatten_pointer(thin: *mut [T; N]) -> *mut Self {
        thin as *mut Self
    }
}

/// Represents the result of a [`BigBox::take`] operation.
///
/// See that method for more detailed documentation.
#[allow(missing_docs)]
pub enum BoxCapture<T: ?Sized, const N: usize, const A: usize> {
    Boxed(Box<T>),
    Inline(InlineBox<T, N, A>),
}

/// A box which stores its [(small)] value inline.
///
/// This is identical to a [`BigBox`] which is obligated to store its contents inline.
///
/// # Layout
///
/// The in-memory layout of this type should not be relied upon,
/// and should be considered unstable.
#[cfg_attr(not(doc), repr(transparent))]
pub struct InlineBox<T: ?Sized, const N: usize, const A: usize> {
    inner: ManuallyDrop<BigBox<T, N, A>>,
}

impl<T: ?Sized, const N: usize, const A: usize> InlineBox<T, N, A> {
    /// # Safety
    ///
    /// The `BigBox` must have its value stored inline.
    unsafe fn from_big_box(big_box: BigBox<T, N, A>) -> Self {
        debug_assert!(big_box.is_inline());
        Self {
            inner: ManuallyDrop::new(big_box),
        }
    }

    /// Attempts to construct an [`InlineBox`], failing if the value is not [small].
    ///
    /// This function will never allocate,
    /// instead it will return an `Err` variant which the passed value can be recovered from.
    ///
    /// [small]: BigBox#which-values-can-be-stored-inline
    pub fn try_new<U>(v: U) -> Result<Self, U>
    where
        T: FromSized<U>,
    {
        if size_of::<U>() > N || align_of::<U>() > 8 {
            return Err(v);
        }

        // NB: `boxed` is null, but has the metadata of `T`.
        let boxed = T::fatten_pointer(ptr::null_mut::<U>());
        let mut inst = BigBox::uninit_from_boxed(boxed);

        // SAFETY: This write is safe bceause:
        // * `ptr` is non-null and live because it points into the `inline`
        //   field of the stack-allocated `inst` object.
        // * `ptr` is derefenceable because `T` is not bigger than `inline`.
        // * `ptr` is correctly aligned because:
        //   - `inst` is aligned to 8 bytes.
        //   - Because of `repr(C)`, the same is true for the `inline` field.
        //   - `T` must be aligned to no more than 8 bytes.
        let ptr: *mut U = ptr::addr_of_mut!(inst.inline).cast();
        unsafe { ptr.write(v) };

        // SAFETY: `inst` is constructed inline:
        // * `boxed` is explicitly null with the metadata for type `U`.
        // * `inline` has been written to with an accurate value of type `U`.
        Ok(unsafe { Self::from_big_box(inst) })
    }

    /// Reads the value from behind the reference.
    ///
    /// # Safety
    ///
    /// As a safety measure,
    /// the type behind the reference must implement [`FromSized`] over the type being read;
    /// this is the same contract imposed by [`BigBox::new`]
    /// but is not enough to ensure correctness.
    ///
    /// The caller must uphold that the reference points to a value of the type being read.
    pub unsafe fn read_unchecked<U>(mut self) -> U
    where
        T: FromSized<U>,
    {
        // SAFETY:
        // * `Take` contract upholds the backing value is not used again.
        // * The caller contract ensures the cast is valid.
        let read = unsafe { ptr::read(self.inner.inline_ptr_mut().cast()) };
        mem::forget(self);
        read
    }
}

impl<T: ?Sized, const N: usize, const A: usize> AsRef<T> for InlineBox<T, N, A> {
    fn as_ref(&self) -> &T {
        // SAFETY: The type's contract ensures the value is always stored inline.
        unsafe { self.inner.inline_ptr().as_ref().unwrap_unchecked() }
    }
}
impl<T: ?Sized, const N: usize, const A: usize> AsMut<T> for InlineBox<T, N, A> {
    fn as_mut(&mut self) -> &mut T {
        // SAFETY: The type's contract ensures the value is always stored inline.
        unsafe { self.inner.inline_ptr_mut().as_mut().unwrap_unchecked() }
    }
}
impl<T: ?Sized, const N: usize, const A: usize> Drop for InlineBox<T, N, A> {
    fn drop(&mut self) {
        // SAFETY:
        // * The type's contract ensures the value is always stored inline.
        // * Since this is the tail of `Drop::drop`, the value behind it is never used again.
        unsafe { self.inner.inline_ptr_mut().drop_in_place() };
    }
}

impl<T: ?Sized, const N: usize, const A: usize> From<InlineBox<T, N, A>> for BigBox<T, N, A> {
    fn from(mut value: InlineBox<T, N, A>) -> Self {
        // SAFETY: `value` is immediately forgotten.
        let big_box = unsafe { ManuallyDrop::take(&mut value.inner) };
        mem::forget(value);
        big_box
    }
}

impl<T: ?Sized, const N: usize, const A: usize> BigBox<T, N, A> {
    fn uninit_from_boxed(boxed: *mut T) -> Self {
        Self {
            inline: UnsafeCell::new([MaybeUninit::uninit(); N]),
            boxed,
        }
    }

    /// Creates a new, unconditionally boxed [`BigBox`].
    #[inline]
    pub fn new_boxed(boxed: Box<T>) -> Self {
        Self::uninit_from_boxed(Box::into_raw(boxed))
    }

    /// Creates a new [`BigBox`], storing [small] values inline.
    ///
    /// [small]: Self#which-values-can-be-stored-inline
    pub fn new<U>(v: U) -> Self
    where
        T: FromSized<U>,
    {
        match InlineBox::try_new(v) {
            Ok(inline) => inline.into(),
            Err(v) => {
                // NB: we'd like to just call `Self::new_boxed(Box::new(v) as Box<T>)`,
                // but this kind of cast doesn't work for any `T`, so we have to do it manually.

                let raw = Box::into_raw(Box::new(v));
                let fat = T::fatten_pointer(raw);
                Self::uninit_from_boxed(fat)
            }
        }
    }

    /// Returns whether this `BigBox` is inline, rather than boxed.
    pub fn is_inline(&self) -> bool {
        self.boxed.is_null()
    }

    /// When the data is [inline], returns a valid-for-reads pointer to the data.
    ///
    /// [inline]: Self::is_inline
    fn inline_ptr(&self) -> *const T {
        // By definition, the data is inline
        let data_start = self.inline.get();
        copy_metadata(self.boxed, data_start.cast())
    }

    /// When the data is [inline], returns a valid-for-writes pointer to the data.
    ///
    /// [inline]: Self::is_inline
    fn inline_ptr_mut(&mut self) -> *mut T {
        // By definition, the data is inline
        let data_start = self.inline.get_mut().as_mut_ptr();
        copy_metadata(self.boxed, data_start.cast()).cast_mut()
    }

    /// Returns this `BigBox` as a `Box` of the same type.
    ///
    /// Note that this method will allocate if the `BigBox` is stored inline.
    pub fn into_boxed(self) -> Box<T> {
        match self.take() {
            BoxCapture::Boxed(boxed) => boxed,
            BoxCapture::Inline(inline) => take_into_box(inline),
        }
    }

    /// Consumes a [`BigBox`], returning the boxed form if it can, and calling `take_inline` on inline values.
    pub fn take(self) -> BoxCapture<T, N, A> {
        let capture = if self.is_inline() {
            // SAFETY: The value is inline.
            let inline = unsafe { InlineBox::from_big_box(self) };
            BoxCapture::Inline(inline)
        } else {
            // SAFETY: When the value is not inline,
            // `boxed` is always constructed by `Box::into_raw`
            // and `self` is never used again
            // (including via drop, since `self` is forgotten).
            let boxed = unsafe { Box::from_raw(self.boxed) };
            mem::forget(self);
            BoxCapture::Boxed(boxed)
        };
        capture
    }
}

impl<T: ?Sized, const N: usize, const A: usize> AsRef<T> for BigBox<T, N, A> {
    fn as_ref(&self) -> &T {
        let data_ptr = self
            .is_inline()
            .then(|| self.inline_ptr())
            .unwrap_or(self.boxed);
        // SAFETY:
        // * `data_ptr` returns a valid pointer to the data,
        //   with the right metadata.
        // * The lifetime of the return type is constrained by the reference to `self`,
        //   so no use-after-free can occur.
        unsafe { data_ptr.as_ref().unwrap_unchecked() }
    }
}

impl<T: ?Sized, const N: usize, const A: usize> AsMut<T> for BigBox<T, N, A> {
    fn as_mut(&mut self) -> &mut T {
        let data_ptr = self
            .is_inline()
            .then(|| self.inline_ptr_mut())
            .unwrap_or(self.boxed);
        // SAFETY:
        // * `inline_ptr_mut` returns a valid pointer to the data,
        //   with the right metadata.
        // * The lifetime of the return type is constrained by the reference to `self`,
        //   so no use-after-free can occur.
        unsafe { data_ptr.as_mut().unwrap_unchecked() }
    }
}

impl<T: ?Sized, const N: usize, const A: usize> Drop for BigBox<T, N, A> {
    fn drop(&mut self) {
        // NB: this effectively the same as just using `self.take()`
        // but is simpler than jumping through `ManuallyDrop` hoops.
        if self.is_inline() {
            // SAFETY:
            // * `inline_ptr_mut` returns a valid pointer to the data,
            //   with the right metadata (incl. drop function).
            // * Since this is the last thing done in `Drop::drop`,
            //   the value is never used again.
            unsafe { self.inline_ptr_mut().drop_in_place() }
        } else {
            // SAFETY:
            // * `inline_ptr_mut` returned `None`, so `boxed` is a valid box.
            // * Since this is the last thing done in `Drop::drop`,
            //   the value is never used again.
            unsafe { drop(Box::from_raw(self.boxed)) }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::fmt::Debug;
    use std::{
        any::{Any, TypeId},
        array::from_fn,
        cell::Cell,
        marker::PhantomData,
        sync::{Arc, Mutex},
        thread::JoinHandle,
    };

    use crate::*;

    trait Anything: 'static + Debug {
        fn as_any(&self) -> &dyn Any;
        fn into_any(self: Box<Self>) -> Box<dyn Any>;

        fn typeid(&self) -> TypeId;
    }

    impl_from_sized_for_trait_object!(dyn Anything);

    impl<T: 'static + Debug> Anything for T {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn into_any(self: Box<Self>) -> Box<dyn Any> {
            self
        }

        fn typeid(&self) -> TypeId {
            TypeId::of::<Self>()
        }
    }

    trait AnySync: Anything + Send + Sync {}
    impl<T: Anything + Send + Sync> AnySync for T {}

    impl_from_sized_for_trait_object!(dyn AnySync);

    fn assert_roundtrip<T: Anything + PartialEq + Debug + Clone>(v: T) {
        let dv = BigBox::<dyn Anything, 64, 8>::new(v.clone());
        let v2: &T = dv.as_ref().as_any().downcast_ref().unwrap();
        assert_eq!(v2, &v);

        let boxed = dv.into_boxed();
        println!("{boxed:?} vs {}", std::any::type_name::<T>());
        println!("{:?} vs {:?}", boxed.typeid(), TypeId::of::<T>());

        let v2: T = *boxed.into_any().downcast().unwrap();
        assert_eq!(v2, v);
    }

    #[test]
    fn roundtrip() {
        assert_roundtrip(());
        assert_roundtrip(PhantomData::<&()>);
        assert_roundtrip(u8::MIN);
        assert_roundtrip(u8::MAX);
        assert_roundtrip(usize::MIN);
        assert_roundtrip(usize::MAX);
        assert_roundtrip(u128::MIN);
        assert_roundtrip(u128::MAX);

        assert_roundtrip([110_u8; 1000]);
    }

    #[test]
    fn cell() {
        let boxed: BigBox<dyn Anything, 64> = BigBox::new(Cell::new(5_u8));

        let a: &Cell<u8> = boxed.as_ref().as_any().downcast_ref().unwrap();
        let b: &Cell<u8> = boxed.as_ref().as_any().downcast_ref().unwrap();

        a.set(10);
        assert_eq!(a.get(), 10);
        assert_eq!(b.get(), 10);
        b.set(12);
        assert_eq!(a.get(), 12);
        assert_eq!(b.get(), 12);
    }

    #[test]
    fn mutex() {
        let mtx = Mutex::new(10_usize);
        let boxed: BigBox<dyn AnySync, 64> = BigBox::new(mtx);
        assert!(boxed.is_inline());
        let arc = Arc::new(boxed);

        let threads: [JoinHandle<()>; 10] = from_fn(|i| {
            let arc = arc.clone();
            std::thread::spawn(move || {
                let mut num = arc
                    .as_ref()
                    .as_ref()
                    .as_any()
                    .downcast_ref::<Mutex<usize>>()
                    .unwrap()
                    .lock()
                    .unwrap();
                *num += i;
            })
        });

        threads.into_iter().for_each(|td| td.join().unwrap());

        let boxed = Arc::into_inner(arc).unwrap();
        assert_eq!(
            *boxed
                .into_boxed()
                .into_any()
                .downcast::<Mutex<usize>>()
                .unwrap()
                .get_mut()
                .unwrap(),
            55,
        );
    }
}
