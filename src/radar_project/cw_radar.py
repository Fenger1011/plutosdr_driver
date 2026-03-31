import adi
import numpy as np
from driver.hardware_setup import *


# Project definitions
uri = "ip:192.168.2.1"
sample_rate = 5e6

def main():
    # Create SDR object
    sdr = create_pluto(uri=uri, sample_rate=sample_rate)

    # Configure receiver
    configure_rx(
        sdr,
        
    )