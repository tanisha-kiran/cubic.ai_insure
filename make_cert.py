"""
Create a self-signed HTTPS certificate for the demo, with no internet and no OpenSSL needed.

    python make_cert.py            # writes data/certs/cert.pem and key.pem
    python make_cert.py --force    # replace an existing one

The certificate covers localhost, 127.0.0.1, this machine's name and its LAN addresses, so other
devices can open https://<this-pc-ip>:8443. Browsers will warn once ("not trusted") because no public
authority signed it: choose "Advanced -> Continue". To remove the warning on a device, import cert.pem
into its trusted certificates.
"""
import argparse
import datetime
import ipaddress
import os
import socket

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CERT_DIR = os.path.join(os.environ.get("QUIETLY_DATA", "data"), "certs")


def lan_addresses():
    found = {"127.0.0.1"}
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            found.add(ip)
    except OSError:
        pass
    try:  # the address used for the default route (no packet is actually sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            found.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(found)


def make_cert(force=False):
    cert_path, key_path = os.path.join(CERT_DIR, "cert.pem"), os.path.join(CERT_DIR, "key.pem")
    if os.path.exists(cert_path) and os.path.exists(key_path) and not force:
        return cert_path, key_path
    os.makedirs(CERT_DIR, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cubic.ai (offline demo)")])
    sans = [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())]
    sans += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in lan_addresses()]
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    cert, key = make_cert(ap.parse_args().force)
    print(f"certificate: {cert}\nkey:         {key}\ncovers: {', '.join(lan_addresses())}")
