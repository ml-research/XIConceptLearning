import gpustat
import subprocess
import time
import argparse

# example call:
#python3 schedule_process.py -g "0,1,2" -f /repositories/tasks/scheduleT1.txt -d "docker exec <containername> zsh -c"
# with .txt containing multiple lines of:
# cd <workingdir> && python scripts/run.py --epochs 42

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-f', '--filename', type=str, required=True)
parser.add_argument('-d', '--docker', type=str, required=True)
parser.add_argument('-w', '--wait', type=int, default=30)
parser.add_argument('-g', '--gpus', type=str, default='', help='available gpu ids seperated by ,')
args = parser.parse_args()

# read process list
file_processes = open(args.filename, 'r')
todo_process_list = [f.replace("\n", "") for f in file_processes.readlines() if f[0] != '#']

# set available_gpus
available_gpus = [int(g) for g in args.gpus.split(',')]

# try as long processes are left
while len(todo_process_list) > 0:
    # read current gpu workload/processes
    gpus = gpustat.new_query()

    # check if there is a free gpu and run the process
    # if no gpu is available wait and retry
    remaining_time_global = 100000000
    for gpu_id, gpu in enumerate(gpus):
        if len(available_gpus) == 0 or gpu_id in available_gpus:
            processes = gpu['processes']
            if len(processes) == 0:
                # gpu is free -> start process
                run_command = "{} \' ".format(args.docker) + todo_process_list.pop(0).replace(
                                     "python", "CUDA_VISIBLE_DEVICES={} python".format(gpu_id)
                                 ) + " \'"
                subprocess.Popen(run_command, shell=True,
                                 stdout=open("./logs/{}_stdout_{}_gpuid{}.log".format(args.filename.split('/')[-1].replace('.txt', ''),
                                                                                      time.time(), gpu_id), "w"),
                                 stderr=open("./logs/{}_stderr_{}_gpuid{}.log".format(args.filename.split('/')[-1].replace('.txt', ''),
                                                                                      time.time(), gpu_id), "w"))

                print('Process started on gpu {}'.format(gpu_id))
            else:
                remaining_time = 0
                for p in processes:
                    command = p['command']  # eg @PS_MCM_GPT2#0d:14:50:58
                    # print(gpu_id, command)
                    # check if rtpt is used
                    if command[0] == '@' and 'first_epoch' not in command:
                        remaining_time_p = command.split('#')[1] \
                            .replace('d', '') \
                            .replace('h', '') \
                            .replace('m', '') \
                            .replace('s', '') \
                            .split(':')
                        days = int(remaining_time_p[0])
                        hours = days * 24 + int(remaining_time_p[1])
                        minutes = hours * 60 + int(remaining_time_p[2])
                        seconds = minutes * 60 + int(remaining_time_p[3])
                        if seconds > remaining_time:
                            remaining_time = seconds

                # check again as soon one gpu is free
                remaining_time_global = min(remaining_time_global, remaining_time)

    # add 3 minutes as buffer
    remaining_time_global += 60 * 3

    # done
    if len(todo_process_list) == 0:
        exit()

    # delay next try for x mins
    delay = min(remaining_time_global, 60 * args.wait)
    print("Next try in {}s".format(delay))
    time.sleep(delay)
