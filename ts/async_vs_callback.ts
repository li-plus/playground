function taskWithCallback(callback: (result: number) => void) {
    const workload = Math.random() * 1000
    console.log(`task with callback started`);
    setTimeout(() => callback(workload), workload)
}

function asyncTask(): Promise<number> {
    const workload = Math.random() * 1000
    return new Promise((resolve, reject) => {
        setTimeout(() => resolve(workload), workload)
    })
}

async function main() {
    console.log('async task started')
    const workload = await asyncTask();
    console.log(`async task finished within ${workload.toFixed(2)} ms`);

    taskWithCallback((result) => {
        console.log(`task with callback finished within ${result.toFixed(2)} ms`);
    });
}

main()
