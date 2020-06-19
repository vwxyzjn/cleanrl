# function delay(n) {  
#   n = n || 2000;
#   return new Promise(done => {
#     setTimeout(() => {
#       done();
#     }, n);
#   });
# }

# var sections = document.querySelectorAll('.report-section')
# for (var index = 0; index < sections.length; index++) {
#     sections[index].querySelector('.wbic-ic-overflow').click()
#     await delay(200)
#     document.querySelectorAll('.ui.borderless.vertical.menu .item')[2].click()
#     await delay(200)
#     document.querySelectorAll('.ui.borderless.vertical.popup-submenu.menu .item')[1].click()
#     await delay(2000)
#     document.querySelectorAll('.ui.primary.button')[document.querySelectorAll('.ui.primary.button').length-1].click()
#     await delay(1000)
# }

import subprocess
import time
import os
import psutil

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


width = 600
height = 600
port = "8086"
url_template = ""
server_proc = subprocess.Popen(
    ["python", "-m", "http.server", port], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

import glob, os
os.chdir(".")
for file in glob.glob("*.svg"):
    new_file = file.replace(" ", "").replace("&", "")
    os.rename(file, new_file)
    url = f"http://localhost:9000/api/render?url=http://localhost:{port}/{new_file}&pdf.landscape=true&pdf.height={height}&pdf.width={width}"
    proc = subprocess.Popen(
        ["curl", "-o", new_file.rstrip(".svg")+".pdf", url], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


time.sleep(5)
kill(server_proc.pid)