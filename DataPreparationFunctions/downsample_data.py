from scipy.signal import decimate
def downsample_data(sample_rate, target_sample_rate, signal, manual_fiducials):
    if target_sample_rate is not None:
            factor = sample_rate // target_sample_rate
            if factor<1:
                raise Exception("Target sample rate is larger than sample rate. Implement upsamling if necessary.")
            else:
                signal = decimate(signal, factor, ftype='iir')
                manual_fiducials = manual_fiducials//factor
    else: 
        signal = None
        manual_fiducials = None
    return signal, manual_fiducials