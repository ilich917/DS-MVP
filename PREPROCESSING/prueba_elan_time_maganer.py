from elan_time_manager import elan_time_manager as etm


file = 'csvs/MesesLSM.csv'

manager = etm(file)

ti, tf = manager.main()

for row in zip(ti,tf):
    print(row[0],row[1])