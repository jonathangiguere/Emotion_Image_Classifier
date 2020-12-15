from google_drive_downloader import GoogleDriveDownloader as gdd
import os

# list of file ids in google drive to get
rar_file_list = ['13ZzvB0D6Ds63wMdpWOqEojmQoIdxDLO2', '1wVujP8pxmm4IYSVPS7U3s0cgeI953_Ld', '1ZSzZSOHUGZ-k2uyKB29PAbt31fhjbpq_',
                 '1P7RvIb5xkt8ni_Tv4v0uvRretyRkq9jZ', '1AtjqP8_8EhImJavUT8jMvWX0xrqI-V_1', '1MZMZkQUVJ2b5PJx2rJpeEq0d_GJWfsLa',
                 '1anY-NbMCODDndIrhQzc6S1CMC5lAL2uz', '10ZKCrQp9w0Ez1A8lLQuK9UHliXoGL_m8', '18-WgjcfXUPlsSSrIa1LAvv-RxEAsQYWS',
                 '1sF4uqspcmZNuKKMJUy1wbyOsKIPEwVeX', '1AXR3TwtwgsTbee8OewfPRfnJXyXj7tFc']

# Loop through compressed rar files
part = 1
for rar_file in rar_file_list:

    # Get the compressed rar files on VM
    gdd.download_file_from_google_drive(file_id=rar_file,
                                        dest_path='./data/part_' + str(part) + '.rar',
                                        unzip=False)

    # Unzip each rar file into the same directory
    # Requires installing unar on vm: https://zoomadmin.com/HowToInstall/UbuntuPackage/unar
    os.system('unar -r ./data/part_' + str(part) + '.rar')

    # Increment variable for output files
    part += 1

