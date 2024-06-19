function signal = normalizeLoudness(signal, fs, targetLoudness)
    loudness = integratedLoudness(signal, fs);
    while abs(loudness - targetLoudness) > 0.001
        gain = 10^((targetLoudness - loudness)/20);
        signal = signal .* gain;
        loudness = integratedLoudness(signal, fs);
    end
end