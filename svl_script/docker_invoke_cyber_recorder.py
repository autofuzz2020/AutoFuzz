import os
import time
folder_ind = 0

user_name = os.popen("whoami").read().strip()
print('user_name', user_name)

docker_name = "apollo_dev_"+user_name
cmd = "docker ps | grep '"+docker_name+"' | awk '{print$1}'"
docker_id = os.popen(cmd).read().strip()
print('docker_id', docker_id)


cmd2 = "docker exec -u "+user_name+" "+docker_id+" /bin/bash -c \"source /apollo/cyber/setup.bash && "+"mkdir "+str(folder_ind)+" && cyber_recorder record -a -o "+str(folder_ind)+"/tmp.record"+"\""
os.popen(cmd2)
print('cmd2', cmd2)

time.sleep(5)

cmd3 = "docker top "+docker_id+" | grep cyber_recorder | awk '{print$2}'"
print('cmd3', cmd3)
pid = os.popen(cmd3).read().strip()
print('pid', pid)

cmd4 = "kill -9 "+str(pid)
os.popen(cmd4)
