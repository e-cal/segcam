# SegCam

Raspberry-pi based camera that lets you segment and object track photo & video objects.

## Setup

1. Flash sd card with nixos sd image
    - [Download latest image](https://hydra.nixos.org/job/nixos/release-24.05/nixos.sd_image.aarch64-linux)
        - or the one I used for guarunteed compatability: [nixos-sd-image-24.05.5596.d51c28603def-aarch64-linux.img](https://hydra.nixos.org/build/274691934)
    - if its a `.zst`, decompress with `unzstd <img-name>.img.zst` (`nix-shell -p zstd --run "unzstd <img-name>.img.zst`)
    - flash the sd card: `sudo dd if=<nixos img> of=<sd card path>`
        - find the sd card path with `lsblk` or watch `dmesg` as you plug it in
2. Boot raspberry pi with the sd card
3. Connect to internet: (as root) `wpa_supplicant -B -i wlan0 -c <(wpa_passphrase 'SSID' 'password') &`
    - _just replace the SSID and password, don't omit the `&` or other symbols_
    - when the command finishes check you're online with `host nixos.org` or `ping 8.8.8.8`
    - if you typo, `pkill wpa_supplicant` and try again
4. [Optional] update raspberry pi firmware
    ```bash
    nix-shell -p raspberrypi-eeprom
    mount /dev/disk/by-label/FIRMWARE /mnt
    BOOTFS=/mnt FIRMWARE_RELEASE_STATUS=stable rpi-eeprom-update -d -a
    ```
5. Generate default configs: `nixos-generate-config`
6. Add nixos hardware channel
    ```
    sudo nix-channel --add https://github.com/NixOS/nixos-hardware/archive/master.tar.gz nixos-hardware
    sudo nix-channel --update
    ```
7. Clone the repo: `nix-shell -p git --run "git clone https://github.com/e-cal/segcam.git"`
8. Install: `cd segcam/system && ./install`
9. Rebuild nixos: `sudo nixos-rebuild boot`
10. Reboot
    - default user is `segcam` with password `segcam`
    - will need to resetup network connection using `networkmanager`
        - list networks: `nmcli dev wifi`
        - connect: `nmcli dev wifi con 'SSID' --ask`
