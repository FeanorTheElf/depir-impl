use std::cell::{RefCell, Cell};
use std::collections::VecDeque;
use std::ffi::{c_void, OsString};
use std::os::windows::prelude::OsStrExt;
use std::time::Instant;
use windows_sys::Win32::Foundation::{HANDLE, GetLastError, GENERIC_READ, WAIT_OBJECT_0, CloseHandle, INVALID_HANDLE_VALUE, WAIT_FAILED, WIN32_ERROR};
use windows_sys::Win32::System::IO::{OVERLAPPED, OVERLAPPED_0, OVERLAPPED_0_0};
use windows_sys::Win32::Storage::FileSystem::{ReadFileEx, CreateFileW, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, FILE_FLAG_NO_BUFFERING};
use windows_sys::Win32::System::Threading::{SetEvent, CreateEventW, WaitForSingleObjectEx, INFINITE};
use append_only_vec::AppendOnlyVec;

extern crate windows_sys;

///
/// The minimal amount of bytes we can read from an SSD.
/// Determined by hardware.
/// 
const SECTOR_SIZE: usize = 4096;

const BUFFER_LEN: usize = SECTOR_SIZE / std::mem::size_of::<ReadType>();

///
/// How many Readtasks we create, i.e. how many read request we
/// pass on to the OS concurrently
/// 
const READTASK_COUNT: usize = 48;

type ReadType = u16;

///
/// A task to read from a queue of read requests from a single file.
/// 
pub struct ReadTask<'context, F>
    where F: FnMut(ReadType, usize)
{
    buffer: Box<[ReadType]>,
    callback: &'context RefCell<F>,
    current_target_index: usize,
    current_source_index: usize,
    event: HANDLE,
    file: HANDLE,
    overlapped: OVERLAPPED,
    // number of currently unfinished tasks
    final_callbacks_counter: &'context Cell<usize>,
    indices: &'context RefCell<VecDeque<(usize, u64)>>
}

impl<'context, F> ReadTask<'context, F>
    where F: FnMut(ReadType, usize)
{
    fn new(event: HANDLE, file: HANDLE, indices: &'context RefCell<VecDeque<(usize, u64)>>, callback: &'context RefCell<F>, final_callbacks_counter: &'context Cell<usize>) -> Self {
        Self {
            buffer: (0..SECTOR_SIZE).map(|_| 0).collect::<Vec<_>>().into_boxed_slice(),
            callback,
            current_target_index: 0,
            current_source_index: 0,
            event: event,
            file: file,
            overlapped: OVERLAPPED { Internal: 0, InternalHigh: 0, Anonymous: OVERLAPPED_0 { Anonymous: OVERLAPPED_0_0 { Offset: 0, OffsetHigh: 0 } }, hEvent: 0 },
            indices: indices,
            final_callbacks_counter: final_callbacks_counter
        }
    }

    unsafe fn callback(&mut self) {
        {
            (*self.callback).borrow_mut()(self.buffer[self.current_source_index % BUFFER_LEN], self.current_target_index as usize);
        }
        let next_indices = {
            (*self.indices).borrow_mut().pop_front()
        };
        if let Some((target_index, source_index)) = next_indices {
            self.run(target_index, source_index);
        } else {
            if (*self.final_callbacks_counter).update(|x| x - 1) == 0 {
                let result = SetEvent(self.event);
                if result == 0 {
                    panic!("SetEvent failed (error code {})", GetLastError());
                }
            }
        }
    }

    ///
    /// Performs the read requests from `indices` until it is empty.
    /// 
    unsafe fn run(&mut self, target_index: usize, source_index: u64) {
    
        self.current_source_index = source_index as usize;
        self.current_target_index = target_index;
        let byte_index = self.current_source_index * std::mem::size_of::<ReadType>();
        let sector_index = byte_index / SECTOR_SIZE;
        let sector_byte_index = sector_index * SECTOR_SIZE;

        self.overlapped.Anonymous.Anonymous.Offset = (sector_byte_index % (1 << 32)) as u32;
        self.overlapped.Anonymous.Anonymous.OffsetHigh = (sector_byte_index / (1 << 32)) as u32;
        self.overlapped.hEvent = self as *mut _ as isize;

        let result = ReadFileEx(
            self.file, 
            self.buffer.as_mut_ptr() as *mut c_void, 
            SECTOR_SIZE as u32, 
            &mut self.overlapped as *mut OVERLAPPED, 
            Some(callback_routine::<F>)
        );
        if result == 0 {
            panic!("ReadFileEx failed (error code {})", GetLastError());
        }
    }
}

unsafe extern "system" fn callback_routine<F>(dwerrorcode: u32, _dwnumberofbytestransfered: u32, lpoverlapped: *mut OVERLAPPED)
    where F: FnMut(ReadType, usize)
{
    if dwerrorcode != 0 {
        panic!("ReadFileEx callback called with error (error code {})", dwerrorcode);
    }
    let task_ptr = lpoverlapped.as_ref().unwrap().hEvent;
    (task_ptr as *mut ReadTask<F>).as_mut().unwrap().callback();
}

///
/// Manages multiple queues of read requests to disk that are performed asynchronously by the OS.
/// 
pub struct DiskReadContext<'env, 'context, 'tasks> {
    // all the boxes are necessary to ensure that pointers stay constant
    files: &'context AppendOnlyVec<(isize, isize, Box<RefCell<VecDeque<(usize, u64)>>>, Box<RefCell<Box<dyn 'env + FnMut(ReadType, usize)>>>, Box<Cell<usize>>)>,
    read_tasks: &'tasks RefCell<Vec<Vec<Box<ReadTask<'context, Box<dyn 'env + FnMut(ReadType, usize)>>>>>>,
    total_read_counter: &'context Cell<usize>
}

#[derive(Clone, Copy)]
pub struct FileReader<'env, 'context, 'tasks, 'parent> {
    current_index: usize,
    parent: &'parent DiskReadContext<'env, 'context, 'tasks>,
    parent_file_index: usize
}

impl<'env, 'context, 'tasks, 'parent> FileReader<'env, 'context, 'tasks, 'parent> {

    pub fn submit(&mut self, location: u64) {
        self.parent.submit_to_file(self.parent_file_index, self.current_index, location);
        self.current_index += 1;
    }

    pub fn wait_for_file_finished(&self) {
        unsafe {
            self.parent.wait_for_processing_finished(self.parent_file_index).unwrap();
        }
    }

    pub fn current_index(&self) -> usize {
        self.current_index
    }
}

impl<'env, 'context, 'tasks> DiskReadContext<'env, 'context, 'tasks> {

    unsafe fn new(
        files: &'context AppendOnlyVec<(isize, isize, Box<RefCell<VecDeque<(usize, u64)>>>, Box<RefCell<Box<dyn 'env + FnMut(ReadType, usize)>>>, Box<Cell<usize>>)>,
        tasks: &'tasks RefCell<Vec<Vec<Box<ReadTask<'context, Box<dyn 'env + FnMut(ReadType, usize)>>>>>>,
        total_read_counter: &'context Cell<usize>
    ) -> Self {
        DiskReadContext { files: files, read_tasks: tasks, total_read_counter }
    }

    pub fn open_file<'parent, F>(&'parent self, filename: &str, callback: F) -> FileReader<'env, 'context, 'tasks, 'parent>
        where F: 'env + FnMut(ReadType, usize)
    {
        unsafe {
            let filename_wide = OsString::from(filename).encode_wide().chain(Some(0)).collect::<Vec<_>>();
            let file = CreateFileW(
                filename_wide.as_ptr() as *const u16, 
                GENERIC_READ, 
                FILE_SHARE_READ, 
                std::ptr::null(), 
                OPEN_EXISTING, 
                FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, 
                std::ptr::null::<()>() as isize
            );
            if file == INVALID_HANDLE_VALUE {
                panic!("CreateFileW failed (error code {})", GetLastError());
            }
            let finish_event = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
            if finish_event == INVALID_HANDLE_VALUE {
                panic!("CreateEventW failed (error code {})", GetLastError());
            }
            let indices = Box::new(RefCell::new(VecDeque::new()));
            let callback = Box::new(RefCell::new(Box::new(callback) as Box<dyn 'env + FnMut(ReadType, usize)>));
            let final_callback_counter = Box::new(Cell::new(0));

            self.files.push((file, finish_event, indices, callback, final_callback_counter));
            self.read_tasks.borrow_mut().push(Vec::new());

            let file_index = self.files.len() - 1;
            return FileReader { current_index: 0, parent: self, parent_file_index: file_index }
        }
    }

    fn submit_to_file(&self, file_index: usize, callback_index: usize, position: u64) {
        let mut read_tasks = self.read_tasks.borrow_mut();
        assert!(read_tasks.len() > file_index);
        assert!(self.files.len() > file_index);
        self.total_read_counter.update(|x| x + 1);
        if read_tasks[file_index].len() < READTASK_COUNT {
            // not yet maximum of readtasks reached - create a new one
            let indices_ptr: &'context RefCell<VecDeque<(usize, u64)>> = &*self.files[file_index].2;
            let callback_ptr: &'context RefCell<Box<dyn 'env + FnMut(ReadType, usize)>> = &*self.files[file_index].3;
            let final_callback_counter: &'context Cell<usize> = &*self.files[file_index].4;
            final_callback_counter.update(|x| x + 1);
            let read_task = ReadTask::new(self.files[file_index].1, self.files[file_index].0, indices_ptr, callback_ptr, final_callback_counter);
            read_tasks[file_index].push(Box::new(read_task));
            unsafe {
                read_tasks[file_index].last_mut().unwrap().run(callback_index, position);
            }
        } else {
            self.files[file_index].2.borrow_mut().push_back((callback_index, position));
        }
    }

    unsafe fn wait_for_processing_finished(&self, file_index: usize) -> Result<(), WIN32_ERROR> {
        let mut read_tasks = self.read_tasks.borrow_mut();
        assert!(read_tasks.len() > file_index);
        assert!(self.files.len() > file_index);
        let finish_event = self.files[file_index].1;

        if read_tasks[file_index].len() == 0 {
            return Ok(());
        }

        let mut wait_state = WaitForSingleObjectEx(finish_event, INFINITE, 1);
        if wait_state == WAIT_FAILED {
            return Err(GetLastError());
        }
        while wait_state != WAIT_OBJECT_0 {
            wait_state = WaitForSingleObjectEx(finish_event, INFINITE, 1);
            if wait_state == WAIT_FAILED {
                return Err(GetLastError());
            }
        }
        // ensure that all tasks are finished
        assert_eq!(self.files[file_index].4.get(), 0);
        read_tasks[file_index].clear();
        return Ok(());
    }
}

///
/// Opens a session in which many read request to disk can be submitted asynchronously.
/// 
/// In theory, this should perform all reads started during `f()` asynchronous in the background,
/// and wait for them to complete before `perform_reads_async` terminates.
/// Currently, (almost) all of them are queued until the end of `f()`, and then performed in parallel.
/// As long as `f()` does not do much computations, this should be equivalent (and we avoid multithreading
/// for now).
/// 
pub fn perform_reads_async<'env, F, T>(f: F) -> T
    where F: for<'a, 'b> FnOnce(&mut DiskReadContext<'env, 'a, 'b>) -> T
{
    // use inner function to more easily debug lifetime mismatches
    unsafe fn internal<'env, 'context, 'tasks, F, T>(
        files: &'context AppendOnlyVec<(isize, isize, Box<RefCell<VecDeque<(usize, u64)>>>, Box<RefCell<Box<dyn 'env + FnMut(ReadType, usize)>>>, Box<Cell<usize>>)>,
        tasks: &'tasks RefCell<Vec<Vec<Box<ReadTask<'context, Box<dyn 'env + FnMut(ReadType, usize)>>>>>>,
        total_read_counter: &'context Cell<usize>,
        f: F
    ) -> T
        where F: FnOnce(&mut DiskReadContext<'env, 'context, 'tasks>) -> T
    {
        let mut readbatch = DiskReadContext::new(files, tasks, total_read_counter);

        let result = f(&mut readbatch);

        let len = readbatch.read_tasks.borrow().len();
        for i in 0..len {
            readbatch.wait_for_processing_finished(i).unwrap();
        }
        return result;
    }
    
    unsafe {
        let files = AppendOnlyVec::new();
        let tasks = RefCell::new(Vec::new());
        let total_read_counter = Cell::new(0);
        let start = Instant::now();

        let result = internal(&files, &tasks, &total_read_counter, f);

        let end = Instant::now();
        println!("Performed {} reads with average speed {} IOPms", total_read_counter.get(), total_read_counter.get() as u128 / (end - start).as_millis());
        assert_eq!(files.len(), tasks.borrow().len());
        assert!(tasks.borrow().iter().all(|tasklist| tasklist.len() == 0));
        for (file, event, _indices, _callback, _final_callback_counter) in files.iter() {
            CloseHandle(*event);
            CloseHandle(*file);
        }
        return result;
    }
}

#[cfg(test)]
use std::panic::catch_unwind;
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::panic::UnwindSafe;

#[cfg(test)]
fn test_with_testfile<F>(testfile: &str, testfile_len: usize, base: F)
    where F: FnOnce() + UnwindSafe
{
    fs::write(testfile, (0..testfile_len).flat_map(|x| (x as u16).to_le_bytes().into_iter()).collect::<Vec<_>>()).unwrap();
    let result = catch_unwind(base);
    fs::remove_file(testfile).unwrap();
    result.unwrap();
}

#[test]
fn test_read_many() {
    let len = 65536;
    test_with_testfile("testfile_test_read_many", len, || {
        let mut actual: Vec<u16> = (0..32).map(|_| 0).collect();
        perform_reads_async(|context| {
            let mut file = context.open_file("testfile_test_read_many", |x, i| actual[i] = x);
            for x in 0..32 {
                file.submit((x * x * x + 7 * x) % len as u64);
            }
        });
        assert_eq!(
            (0..32).map(|x| (x * x * x + 7 * x) % len).map(|x| x as u16).collect::<Vec<_>>(),
            actual
        );
    })
}

#[test]
fn test_read_many_few_reads() {
    let len = 65536;
    test_with_testfile("testfile_test_read_many_few_reads", len, || {
        let mut actual: Vec<u16> = (0..128).map(|_| 0).collect();
        perform_reads_async(|context| {
            let mut file = context.open_file("testfile_test_read_many_few_reads", |x, i| actual[i] = x);
            for x in 0..128 {
                file.submit((x * x * x + 7 * x) % len as u64);
            }
        });
        assert_eq!(
            (0..128).map(|x| (x * x * x + 7 * x) % len).map(|x| x as u16).collect::<Vec<_>>(),
            actual
        );
    })
}


#[test]
fn test_open_file_no_read() {
    let len = 65536;
    test_with_testfile("testfile_test_open_file_no_read", len, || {
        perform_reads_async(|context| {
            context.open_file("testfile_test_open_file_no_read", |_, _| {});
        });
    });
}