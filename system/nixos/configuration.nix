{ config, lib, pkgs, ... }:

{
  imports =
    [
      # NOTE: first run 
      # sudo nix-channel --add https://github.com/NixOS/nixos-hardware/archive/master.tar.gz nixos-hardware
      # sudo nix-channel --update
      <nixos-hardware/raspberry-pi/4>
      # or comment above and uncomment below
      # "${builtins.fetchGit { url = "https://github.com/NixOS/nixos-hardware.git"; }}/raspberry-pi/4"
      ./hardware-configuration.nix # generated by nixos-generate-config
    ];

  boot.loader.grub.enable = false;
  boot.loader.generic-extlinux-compatible.enable = true;

  boot.kernelPackages = pkgs.linuxPackages_rpi4;
  hardware.enableRedistributableFirmware = true;
  hardware.raspberry-pi."4" = {
    fkms-3d.enable = true;
    touch-ft5406.enable = true;
  };
  # services.libinput.enable = true;

  networking.hostName = "segcam";
  networking.networkmanager.enable = true;
  services.openssh.enable = true;

  # Programs etc

  programs.hyprland.enable = true;

  nixpkgs.overlays = [
    (_: super: {
      neovim-custom = pkgs.wrapNeovimUnstable
        (super.neovim-unwrapped.overrideAttrs (oldAttrs: {
          buildInputs = oldAttrs.buildInputs ++ [ super.tree-sitter ];
        })) (pkgs.neovimUtils.makeNeovimConfig {
          extraLuaPackages = p: with p; [ p.magick ];
          extraPython3Packages = p:
            with p; [
              pynvim
              jupyter-client
              ipython
              nbformat
              cairosvg
            ];
          extraPackages = p: with p; [ imageMagick ];
          withNodeJs = true;
          withRuby = true;
          withPython3 = true;
          customRC = "luafile ~/.config/nvim/init.lua";
        });

      pythonWithPkgs = super.python3.withPackages (ps:
        with ps; [
          pip
          setuptools
          wheel
          ipython
          jupyter
          (catppuccin.overridePythonAttrs (oldAttrs: {
            propagatedBuildInputs = (oldAttrs.propagatedBuildInputs or [ ])
              ++ [ pygments ];
          }))
          pygments
          numpy
        ]);
    })
  ];


  environment.systemPackages = with pkgs; [ 
    neovim-custom
    git
    github-cli
    tmux
    htop
    kitty
    chromium

    zoxide
    direnv
    starship
    fzf
    eza
    bat
    ripgrep
    jq
    ffmpeg

    libinput
    gcc
    gnumake
    cmake
    clang
    clang-tools
    tree-sitter
    pythonWithPkgs
    nodejs_22
    lua

    yapf
    shfmt
    nixfmt-classic
    stylua
    nodePackages.prettier

  ];
  qt.enable = true;

  fonts.packages = with pkgs; [
    (nerdfonts.override {
      fonts = [ "JetBrainsMono" "FantasqueSansMono" "FiraCode" ];
    })
    material-icons
    noto-fonts
    liberation_ttf
  ];


  programs.zsh = {
    enable = true;
    autosuggestions.enable = true;
    syntaxHighlighting.enable = true;
  };
  users.defaultUserShell = pkgs.zsh;


  users = {
    mutableUsers = false;
    users.ecal = {
      isNormalUser = true;
      password = "ecal";
      extraGroups = [ "wheel" "networkmanager" ];
    };
  };

  # Might want later:

  # Enable the X11 windowing system.
  # services.xserver.enable = true;
  # Enable the GNOME Desktop Environment.
  # services.xserver.displayManager.gdm.enable = true;
  # services.xserver.desktopManager.gnome.enable = true;
  
  # Configure keymap in X11
  # services.xserver.xkb.layout = "us";
  # services.xserver.xkb.options = "eurosign:e,caps:escape";

  # Enable sound.
  # hardware.pulseaudio.enable = true;
  # OR
  # services.pipewire = {
  #   enable = true;
  #   pulse.enable = true;
  # };

    programs.nix-ld.enable = true;
  programs.nix-ld.libraries = with pkgs; [
    alsa-lib
    at-spi2-atk
    at-spi2-core
    atk
    cairo
    cups
    curl
    dbus
    expat
    fontconfig
    freetype
    fuse3
    gdk-pixbuf
    ghostscript
    glib
    gtk3
    icu
    libgcc.lib
    libGL
    libappindicator-gtk3
    libdrm
    libglvnd
    libnotify
    libpulseaudio
    libunwind
    libusb1
    libuuid
    libxkbcommon
    libxml2
    mesa
    nspr
    nss
    openssl
    pango
    pipewire
    stdenv.cc.cc
    systemd
    vulkan-loader
    webkitgtk
    zlib
  ];


  system.stateVersion = "24.05"; # don't change
}

