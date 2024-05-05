import serial
from datetime import datetime
import time
import subprocess
log_data = 0
def open_serial():
    global ser, log_data
    ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
    ser.reset_input_buffer()
    
    time.sleep(1)
    send_serial("inc_his")
    time.sleep(1)
    log_data = 1
    send_serial("success")

def send_serial(command):
    global ser
    ser.write('{} \n\r'.format(str(command)).encode())

def read_serial():
    global ser, log_data
    while (ser.in_waiting==0):
        pass
     #   print("serial data available")
    try:
        serial_output = ser.readline().decode()
        #print(serial_output)
        if(log_data==1):
            with open("temp_logs/"+datetime.now().strftime("%m%d%Y")+".txt", "a") as myfile:
                    myfile.write(serial_output)
            if 'history_start' in serial_output:
                log_data=1
                print("logging incubator temp now:")
                with open("temp_logs/"+datetime.now().strftime("%m%d%Y")+".txt", "a") as myfile:
                    myfile.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S")+"\n")
            if 'history_stop' in serial_output:
                log_data=0
                print("incubator temp logged")
                
        if 'auto_focus' in serial_output:
            income_serial_command = 'auto_focus'
        elif 'capture' in serial_output:
            income_serial_command = 'capture'
        elif 'cancel' in serial_output:
            income_serial_command = 'cancel'
        elif 'pi_off' in serial_output:
            income_serial_command = 'pi_off'
        elif 'new_sample' in serial_output:
            income_serial_command = 'new_sample'
        elif 'chlorine' in serial_output:
            income_serial_command = 'chlorine'
            
        elif 'time' in serial_output:
            rtc_time=serial_output
            print(rtc_time)
            try:
                dt = datetime.strptime((rtc_time.strip().strip("time=")),'%d%m%Y %H%M%S')
                print(dt)
                subprocess.call(['sudo', 'date', '-s', '{:}'.format(dt.strftime('%Y/%m/%d %H:%M:%S'))], shell=False) #Sets system time (Requires root, obviously)
                time.sleep(1)
            except:
        # Handle the exception if date-time string parsing fails
                print("Failed to parse date-time string. Keeping the system time unchanged.")    
            with open("log.txt", "a") as myfile:
               # now = datetime.now() # current date and time
                myfile.write("Boot: "+datetime.now().strftime("%m/%d/%Y %H:%M:%S")+"\n")
                myfile.close()
        elif 'ID' in serial_output:
            print(serial_output)
            income_serial_command = 'ID'
            sample_ID = serial_output

                
    except UnicodeDecodeError:
        # when arduino serial boots up, it sometimes have error
        print("unicodedecodeerror")
        pass
    return serial_output