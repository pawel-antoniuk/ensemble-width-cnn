inputDir = 'rec_front_small_uniform/spat';
outputDir = 'frontNewMIT3';

filenames = dir(fullfile(inputDir, "*.wav"));
fullFilenames = fullfile(inputDir, {filenames.name});
[~, filenames, ~] = fileparts({filenames.name});
filenameParts = split(filenames, "_");
realWidths = erase(filenameParts(1, :, 4), 'width');
realWidths = str2double(convertCharsToStrings(realWidths));
realLocs = erase(filenameParts(1, :, 6), 'azoffset');
realLocs = str2double(convertCharsToStrings(realLocs));
HRTFIDs = filenameParts(1, :, 2);
HRTFGroupNames = filenameParts(1, :, 3);

if ~isfolder(outputDir)
    mkdir(outputDir)
end

delete(fullfile(outputDir, '*.wav'))

for i = 1:length(fullFilenames)
    HRTFID = HRTFIDs(i);
    HRTFGroupName = HRTFGroupNames(i);
    width = realWidths(i); % 0-90
    loc = realLocs(i); % -180-180
    if HRTFGroupName == "mit" && width + abs(loc) <= 90
        copyfile(fullFilenames{i}, outputDir)
    end
end

