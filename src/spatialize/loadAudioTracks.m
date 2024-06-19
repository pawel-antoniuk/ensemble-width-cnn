% load tracks routine
function [tracks,trackNames] = loadAudioTracks(audioFilename, params)
    songName = fullfile(audioFilename.folder, audioFilename.name);
    trackFilenames = dir(fullfile(songName, '*.wav'));
    audioInfo = audioinfo(fullfile(trackFilenames(1).folder, ...
        trackFilenames(1).name));
    totalSamples = audioInfo.TotalSamples;
    tracks = zeros(totalSamples, length(trackFilenames));

    for iTrackFilename = 1:length(trackFilenames)
        trackPath = fullfile(trackFilenames(iTrackFilename).folder, ...
            trackFilenames(iTrackFilename).name);
        [track,Fs] = audioread(trackPath);

        if Fs ~= params.RecordingsExpectedFs
            error('Track frequency is not expected frequency');
        end

        tracks(:, iTrackFilename) = track;
    end

    trackNames = {trackFilenames.name};
end
