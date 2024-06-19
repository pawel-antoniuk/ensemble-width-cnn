T = 10;
fs = 48000;
nch = 15;
s = wgn(T * fs, nch, 1);
outputDir = ['../recordings-all/Noise' num2str(nch)];

for iChannel = 1:size(s, 2)
    name = sprintf("%s/%d.wav", outputDir, iChannel);
    audiowrite(name, s(:, iChannel), fs, BitsPerSample=32)
end
