import pyshark
from ttictoc import tic,toc

metadata = {}

def append_metadata(number, file_name, creation_date, file_size, target_device, category, sub_category, filter, total_pkts, attack_pkts):
    metadata[number] = {
        'number': number,
        'file_name': file_name,
        'creation_date': creation_date,
        'file_size': file_size,
        'target_device': target_device,
        'category': category,
        'sub_category': sub_category,
        'filter': filter,
        'total_pkts': total_pkts,
        'attack_pkts': attack_pkts,
    }

base_folder = 'C:\\Users\\marce\\git\\iot-threat-classifier\\datasets\\iot_intrusion_dataset'

append_metadata(
    5,
    f"{base_folder}\\mitm-arpspoofing-4-dec.pcap",
    '03/06/2019',
    18813,
    'NUGU',
    'Man in the Middle (MITM)',
    'ARP Spoofing',
    "eth.addr == f0:18:98:5e:ff:9f and (((ip.addr == 192.168.0.24) and !icmp and tcp) or (arp.src.hw_mac == f0:18:98:5e:ff:9f and (arp.dst.hw_mac == 04:32:f4:45:17:b3 or arp.dst.hw_mac == 88:36:6c:d7:1c:56)))",
    19914,
    13211
)

def parse_metadata():
    for key,value in metadata.items():

        print(f'Processing file number {key}...')

        tic()
        cap = pyshark.FileCapture(value['file_name'], display_filter=f"!({value['filter']})")
        benign_packets = []
        for pkt in cap: 
            benign_packets.append(pkt)
        cap.close()
        print('benign_packets',len(benign_packets),toc())
        assert len(benign_packets) == (value['total_pkts'] - value['attack_pkts'])

        tic()
        cap = pyshark.FileCapture(value['file_name'], display_filter=value['filter'])
        malign_packets = []
        for pkt in cap: 
            malign_packets.append(pkt)
        cap.close()
        print('malign_packets',len(malign_packets),toc())
        assert len(malign_packets) == value['attack_pkts']

parse_metadata()