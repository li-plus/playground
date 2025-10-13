async function main() {
  {
    // task with callback
    console.log('task with callback started');
    setTimeout(() => {
      console.log('task with callback finished');
    }, 1000);
  }

  {
    // promise then
    console.log('async task started');
    new Promise((resolve) => setTimeout(resolve, 1000)).then((result) => console.log('async task finished'));
  }

  {
    // best practise: async task
    console.log('async task started');
    await new Promise((resolve) => setTimeout(resolve, 1000));
    console.log('async task finished');
  }
}

main()
