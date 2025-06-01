import os
import xml.etree.ElementTree as ET
from collections import Counter

def find_mode_paths(scan_folder, target_modes):
    mode_paths = {mode: [] for mode in target_modes}
    namespace = {'ns': 'http://baden.varian.com/cr.xsd'}  # XML namespace
    
    for root, _, files in os.walk(scan_folder):
        if 'Scan.xml' in files:
            xml_path = os.path.join(root, 'Scan.xml')
            try:
                tree = ET.parse(xml_path)
                root_element = tree.getroot()
                
                for mode in root_element.findall('.//ns:ModeName', namespaces=namespace):
                    if mode.text in target_modes:
                        mode_paths[mode.text].append(xml_path)
            except ET.ParseError:
                print(f"Error parsing {xml_path}")
            except Exception as e:
                print(f"Unexpected error with {xml_path}: {e}")
    
    return mode_paths

if __name__ == "__main__":
    scan_folder = "jeppes_project/data/scans"
    target_modes = {"Thorax Advanced", "Short Thorax"}
    mode_paths = find_mode_paths(scan_folder, target_modes)
    
    for mode, paths in mode_paths.items():
        print(f"{mode}:")
        for path in paths:
            print(f"  {path}")
