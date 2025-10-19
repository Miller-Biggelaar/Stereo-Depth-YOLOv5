import csv

def log_detection_to_csv(file_path, timestamp, class_name, x, y, depth):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, class_name, x, y, depth])