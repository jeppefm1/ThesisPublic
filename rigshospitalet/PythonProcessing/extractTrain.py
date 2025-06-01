import os
import xml.etree.ElementTree as ET
import csv
from datetime import datetime

def extract_scan_data(scan_folder, output_csv):
    namespace = {'ns': 'http://baden.varian.com/cr.xsd'}  # XML namespace
    data = []
    
    for root, _, files in os.walk(scan_folder):
        if 'Scan.xml' in files:
            xml_path = os.path.join(root, 'Scan.xml')
            try:
                # Extract reconstruction ID from path (last part after splitting by '/')
                reconstruction_id = root.split('/')[-1]
                
                # Parse XML file
                tree = ET.parse(xml_path)
                root_element = tree.getroot()
                
                # Extract required fields
                creation_date = root_element.find('.//ns:CreationDate', namespaces=namespace)
                creation_date_text = creation_date.text if creation_date is not None else 'N/A'
                
                # Extract Scan Type from XML
                scan_type = root_element.find('.//ns:ModeName', namespaces=namespace)
                scan_type_text = scan_type.text if scan_type is not None else 'N/A'
                
                machine_id = root_element.find('.//ns:MachineId', namespaces=namespace)
                machine_id_text = machine_id.text if machine_id is not None else 'N/A'
                
                # Process all acquisitions in this scan
                acquisitions = root_element.findall('.//ns:Acquisition', namespaces=namespace)
                
                for acquisition in acquisitions:
                    acquisition_id = acquisition.get('Id', 'N/A')
                    data.append([
                        reconstruction_id,
                        creation_date_text,
                        scan_type_text,
                        machine_id_text,
                        acquisition_id
                    ])
            except ET.ParseError:
                print(f"Error parsing {xml_path}")
            except Exception as e:
                print(f"Unexpected error with {xml_path}: {e}")
    
     # Sort data based on creation date with exact format: 2023-01-11T09:44:32.166059
    def parse_date(date_str):
        if date_str == 'N/A':
            return datetime.max
        try:
            # Handle microseconds correctly by ensuring the format matches exactly
            # Some dates might have different microsecond precision
            parts = date_str.split('.')
            if len(parts) == 2:
                # Ensure microseconds have proper padding
                microseconds = parts[1].ljust(6, '0')[:6]
                date_str = f"{parts[0]}.{microseconds}"
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            # Fallback for dates without microseconds
            try:
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                print(f"Warning: Could not parse date format for: {date_str}")
                return datetime.max

    # Sort the data based on creation date
    data.sort(key=lambda x: parse_date(x[1]))
    
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Reconstruction ID", "Creation Date", "Scan Type", "Machine ID", "Acquisition ID"])
        writer.writerows(data)
    
    print(f"Data extracted and saved to {output_csv}")


if __name__ == "__main__":
    extract_scan_data("jeppes_project/data/scans", "trainScans.csv")