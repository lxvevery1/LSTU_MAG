#!/usr/bin/env bash
#
# KVM / libvirt hypervisor setup for Arch Linux
# Target: Intel Core i5-2520M (Sandy Bridge), UEFI, single ext4 disk
#
# Run as:  sudo ./setup-kvm.sh
#
set -euo pipefail

# ---------------------------------------------------------------- helpers ---
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m    WARN: %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    OK: %s\033[0m\n' "$*"; }

TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || echo stanik)}"
IMG_DIR="/var/lib/libvirt/images"

# ------------------------------------------------------- 1. sanity checks ---
log "1/7 Preflight checks"

if [[ ! -e /sys/firmware/efi ]]; then
    warn "system booted via Legacy BIOS. UEFI firmware packages are still installed,"
    info "but VMs will use SeaBIOS unless you switch the host to UEFI."
else
    ok "booted via UEFI"
fi

if grep -qw vmx /proc/cpuinfo; then
    ok "VT-x (vmx) is exposed by the CPU"
    VTX_PRESENT=1
else
    warn "VT-x (vmx) NOT exposed in /proc/cpuinfo"
    info "The i5-2520M supports VT-x, but it is switched OFF in the firmware."
    info "Enable it:  reboot -> F2 (Dell) -> Security / CPU Setup"
    info "           -> 'Virtualization' or 'Intel Virtualization Technology' -> Enabled"
    info "Without this, KVM cannot work and VMs will refuse to boot."
    VTX_PRESENT=0
fi

# ------------------------------------------------------------ 2. packages ---
log "2/7 Installing packages"

# `libvirt` is a single consolidated package on current Arch (daemon +
# qemu + network drivers). qemu-desktop gives us the full QEMU incl. SPICE.
PKGS=(
    libvirt
    qemu-desktop
    virt-manager
    virt-viewer
    edk2-ovmf          # UEFI firmware for guests
    swtpm              # software TPM, needed for Windows 11 guests
    dnsmasq            # libvirt's default NAT network
    iptables
    nftables
    intel-ucode       # not installed yet; keeps this old Haswell stable
)

info "installing: ${PKGS[*]}"
pacman -S --needed --noconfirm "${PKGS[@]}"
ok "packages installed"

# ------------------------------------------------------- 3. mkinitcpio hook ---
log "3/7 Adding kvm_intel to the initramfs"

# Not strictly required (kvm loads on demand), but loading it early means the
# hypervisor is available before any early-boot service asks for it, and it
# survives `autodetect` pruning the module list.
MKINITCPIO_CONF=/etc/mkinitcpio.conf
if grep -qE '^MODULES=\(.*kvm_intel' "$MKINITCPIO_CONF"; then
    info "kvm_intel already present in MODULES - skipping initramfs rebuild"
else
    sed -i -E 's/^MODULES=\((.*)\)/MODULES=(\1 kvm_intel)/' "$MKINITCPIO_CONF"
    info "MODULES=($(grep -E '^MODULES=' "$MKINITCPIO_CONF"))"
    mkinitcpio -P >/dev/null
    ok "initramfs regenerated for all kernels"
fi

# ------------------------------------------------------ 4. load kvm module ---
log "4/7 Loading the KVM kernel modules"

if (( VTX_PRESENT )); then
    modprobe kvm_intel
    ok "kvm_intel loaded"
else
    # Modules exist; they just refuse to initialise without VT-x in the CPU.
    modprobe kvm_intel 2>/dev/null || true
    warn "kvm_intel did not load - VT-x is disabled in BIOS."
    info "Re-run this script after enabling VT-x in the firmware."
fi

# --------------------------------------------------- 5. load kvm at boot ----
log "5/7 Persisting the module across reboots"

if [[ -f /etc/modules-load.d/kvm.conf ]]; then
    info "/etc/modules-load.d/kvm.conf already exists"
else
    install -d -m755 /etc/modules-load.d
    printf 'kvm\nkvm_intel\n' > /etc/modules-load.d/kvm.conf
    ok "created /etc/modules-load.d/kvm.conf"
fi

# ------------------------------------------------------ 6. user permissions ---
log "6/7 Granting $TARGET_USER access to libvirt"

if id -nG "$TARGET_USER" | grep -qw libvirt; then
    info "$TARGET_USER is already a member of the libvirt group"
else
    usermod -aG libvirt "$TARGET_USER"
    ok "added $TARGET_USER to group 'libvirt'"
fi

# Polkit rule lets virt-manager manage VMs without a password prompt once the
# user is logged in, provided polkit is running (it is, via dbus).
if [[ ! -f /etc/polkit-1/rules.d/49-libvirt-virt-manager.rules ]]; then
    cat > /etc/polkit-1/rules.d/49-libvirt-virt-manager.rules <<'EOF'
// Allow members of the "libvirt" group to manage system libvirt without a
// password prompt. Follows the packaging guideline for virt-manager.
polkit.addRule(function(action, subject) {
    if (action.id.indexOf("org.libvirt.unix.") === 0 &&
        subject.isInGroup("libvirt")) {
        return polkit.Result.YES;
    }
});
EOF
    ok "installed polkit rule for passwordless libvirt management"
fi

mkdir -p "$IMG_DIR"
# NOTE: Arch's libvirt ships a `libvirt` *group* but no `libvirt` *user* (only
# `libvirt-qemu`), so `chown libvirt:libvirt` fails. The images directory is
# root-owned; libvirtd (running as root) creates the disk files and chowns each
# one to libvirt-qemu itself. We only widen the group so members of `libvirt`
# can list the directory.
if getent group libvirt >/dev/null; then
    chown root:libvirt "$IMG_DIR"
    ok "image directory ready at $IMG_DIR (root:libvirt)"
else
    warn "group 'libvirt' does not exist - leaving $IMG_DIR at package defaults"
fi
chmod 755 "$IMG_DIR"

# --------------------------------------------------------- 7. start daemon ---
log "7/7 Enabling and starting libvirtd"

systemctl enable --now libvirtd.service
ok "libvirtd enabled and running"

# Start the default NAT network (virbr0) so guests get internet + DNS.
if virsh net-list --all | grep -q 'default'; then
    virsh net-start default 2>/dev/null || true
    virsh net-autostart default
    ok "default NAT network started and set to autostart"
else
    warn "default network not found - check 'virsh net-list --all'"
fi

# --------------------------------------------------------------- summary ----
log "Verification"

printf '    kvm module : %s\n' "$(lsmod | grep -q '^kvm_intel' && echo loaded || echo 'NOT loaded')"
printf '    /dev/kvm   : %s\n' "$([[ -e /dev/kvm ]] && echo present || echo 'ABSENT')"
printf '    libvirtd   : %s\n' "$(systemctl is-active libvirtd)"
printf '    virbr0     : %s\n' "$(ip -br addr show virbr0 2>/dev/null || echo 'not up')"
printf '    accel      : %s\n' "$(virsh capabilities 2>/dev/null | grep -oE '<domain type=.kvm.' | head -1 || echo unavailable)"

cat <<EOF

$(printf '\033[1;32mSetup complete.\033[0m')

Next steps:

  1. If VT-x was missing, enable it in BIOS now and reboot, then re-run:
       sudo $0

  2. Log out and back in (group 'libvirt' needs a fresh session), then:
       virsh list --all
       virt-manager

  3. Create your first VM:
       virt-install \\
         --name test \\
         --memory 4096 --vcpus 2 \\
         --disk size=40 \\
         --cdrom ~/Downloads/linux.iso \\
         --network network=default \\
         --graphics spice,listen=127.0.0.1 \\
         --video qxl \\
         --boot uefi

  Images land in $IMG_DIR
EOF
