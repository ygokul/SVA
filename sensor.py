import serial
import csv
import time
import logging

serial_port = 'COM8'  
baud_rate = 9600
csv_file = 'distance_data.csv'
a=0

try:
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    print(f"Connected to {serial_port} at {baud_rate} baud.")
except serial.SerialException as e:
    print(f"Error: {e}")
    exit()

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Distance (cm)'])  

    print("Reading data from Arduino and saving to CSV...")
    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').strip()
                if data.startswith("Distance: "): 
                    distance = data.split(" ")[1]  
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([distance])
                    print(f"Saved: {distance} cm")
    except KeyboardInterrupt:
        print("\nData logging stopped.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()
        print("Serial port closed.")
