import numpy as np
import csv

#refernce
#https://blog.csdn.net/qq_35733521/article/details/87889480

# the feature list
def get_col_types():
    protocol_type = ['icmp', 'tcp', 'udp']
    service_type = ['IRC', 'X11', 'Z39_50', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain',
                    'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher',
                    'hostnames', 'http', 'http_443', 'icmp', 'imap4', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
                    'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp',
                    'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
                    'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i',
                    'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']
    flag_type = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    train_label_type = ['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.',
                        'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'normal.', 'perl.', 'phf.', 'pod.',
                        'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.',
                        'warezmaster.']
    test_label_type = ['apache2.', 'back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'httptunnel.', 'imap.',
                       'ipsweep.', 'land.', 'loadmodule.', 'mailbomb.', 'mscan.', 'multihop.', 'named.', 'neptune.',
                       'nmap.', 'normal.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'processtable.', 'ps.', 'rootkit.',
                       'saint.', 'satan.', 'sendmail.', 'smurf.', 'snmpgetattack.', 'snmpguess.', 'sqlattack.',
                       'teardrop.', 'udpstorm.', 'warezmaster.', 'worm.', 'xlock.', 'xsnoop.', 'xterm.']
    label_type = [['normal.'],
                  ['ipsweep.', 'mscan.', 'nmap.', 'portsweep.', 'saint.', 'satan.'],
                  ['apache2.', 'back.', 'land.', 'mailbomb.', 'neptune.', 'pod.', 'processtable.', 'smurf.', 'teardrop.', 'udpstorm.'],
                  ['buffer_overflow.', 'httptunnel.', 'loadmodule.', 'perl.', 'ps.', 'rootkit.', 'sqlattack.', 'xterm.'],
                  ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'named.', 'phf.', 'sendmail.', 'snmpgetattack.',
                   'snmpguess.', 'spy.', 'warezclient.', 'warezmaster.', 'worm.', 'xlock.', 'xsnoop.']]
    return protocol_type,service_type,flag_type,label_type




# the data_processing function
def handle_data():
    protocol_type,service_type,flag_type,label_type = get_col_types()
    source_file = 'kddcup.data.corrected'
    handled_file = 'train_data_10.csv'  # write to csv file
    data_file = open(handled_file, 'w', newline='')
    csv_writer = csv.writer(data_file)
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        for row in csv_reader:
            row[1] = protocol_type.index(row[1])
            
            row[2] = service_type.index(row[2])
            row[3] = flag_type.index(row[3])
            for labels in label_type:
                if labels.count(row[-1])>0:
                    row[-1] = label_type.index(labels)
            csv_writer.writerow(row)
        data_file.close()
    
    test_source_file = 'corrected'
    test_handled_file = 'test_data.csv'  # write to csv file
    test_data_file = open(test_handled_file, 'w', newline='')
    test_csv_writer = csv.writer(test_data_file)
    with open(test_source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        for row in csv_reader:
            row[1] = protocol_type.index(row[1])
            row[2] = service_type.index(row[2])
            row[3] = flag_type.index(row[3])
            for labels in label_type:
                if labels.count(row[-1]) > 0:
                    row[-1] = label_type.index(labels)
            test_csv_writer.writerow(row)
        test_data_file.close()
    print('pre process completed!')


if __name__ == '__main__':
    handle_data()