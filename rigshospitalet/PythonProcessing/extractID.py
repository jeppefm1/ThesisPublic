import os
import xml.etree.ElementTree as ET
import csv

def extract_scan_data(scan_folder, output_csv):
    namespace = {'ns': 'http://baden.varian.com/cr.xsd'}  # XML namespace
    data = []
    
    for root, _, files in os.walk(scan_folder):
        if 'Scan.xml' in files:
            xml_path = os.path.join(root, 'Scan.xml')
            try:
                tree = ET.parse(xml_path)
                root_element = tree.getroot()
                
                creation_date = root_element.find('.//ns:CreationDate', namespaces=namespace)
                machine_serial = root_element.find('.//ns:MachineSerialNumber', namespaces=namespace)
                machine_id = root_element.find('.//ns:MachineId', namespaces=namespace)
                
                acquisitions = root_element.findall('.//ns:Acquisition', namespaces=namespace)
                
                for acquisition in acquisitions:
                    acquisition_id = acquisition.get('Id', 'N/A')
                    data.append([
                        root,
                        creation_date.text if creation_date is not None else 'N/A',
                        machine_serial.text if machine_serial is not None else 'N/A',
                        machine_id.text if machine_id is not None else 'N/A',
                        acquisition_id
                    ])
            except ET.ParseError:
                print(f"Error parsing {xml_path}")
            except Exception as e:
                print(f"Unexpected error with {xml_path}: {e}")
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Path", "Creation Date", "Machine Serial Number", "Machine ID", "Acquisition ID"])
        writer.writerows(data)
    
if __name__ == "__main__":
    scan_folder = "jeppes_project/data/scans"
    output_csv = "scan_data.csv"  # Output file name
    extract_scan_data(scan_folder, output_csv)
    print(f"Data saved to {output_csv}")


import pandas as pd

df = pd.read_csv("scan_data.csv")
unique_files = df["File Path"].nunique()
print(f"Unique Scan.xml files: {unique_files}")


acquisitions_per_file = df["File Path"].value_counts()
print(acquisitions_per_file[acquisitions_per_file > 1])