import csv
import yaml

# Membaca file CSV
with open('salto.csv', 'r') as file:
    # Membaca isi file sebagai string
    data = file.read()

# Memisahkan data YAML menjadi setiap entri
entries = data.split('---\n')

# Membuka file CSV baru untuk menulis
with open('Salto_fiks.csv', 'a', newline='') as csvfile:
    # Menulis header file CSV baru
    fieldnames = ['x', 'y', 'z']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterasi melalui setiap entri YAML
    for entry in entries:
        # Mengkonversi teks YAML ke struktur data Python
        entry_data = yaml.safe_load(entry)

        # Mengambil nilai x, y, dan z dari orientation dan linear_acceleration
        orientation = entry_data.get('orientation', {})
        #linear_acceleration = entry_data.get('linear_acceleration', {})

        x = orientation.get('x', 0.0)
        y = orientation.get('y', 0.0)
        z = orientation.get('z', 0.0)

        # Menulis nilai ke file CSV
        writer.writerow({'x': x, 'y': y, 'z': z})
