function tracks = normalizeAudioTracks(tracks, params)
    for iTrack = 1:size(tracks,2)
        tracks(:, iTrack) = normalizeLoudness(tracks(:, iTrack), ...
             params.RecordingsExpectedFs, ...
             params.TargetTrackLoudness);
    end
end