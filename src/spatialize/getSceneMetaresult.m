function metaResults = getSceneMetaresult(HRTF, audioName, trackNames, ...
    elevation, azimuthLocation, params)

    width = rand * params.MaxWidth;
%     azimuthOffset = (params.AzimuthOffsetRange(2) ...
%         - params.AzimuthOffsetRange(1)) .* rand ...
%         + params.AzimuthOffsetRange(1);
%     azimuthOffset = wrapTo180(azimuthOffset);
    azimuthOffset = azimuthLocation;

%     randTrackAngles = rand(length(trackNames), 1);
%     randTrackAngles = rescale(randTrackAngles, -width, width);
%     randTrackAngles = randTrackAngles + azimuthOffset;
%     randTrackAngles = wrapTo180(randTrackAngles);
%     randTrackAngles(:, 2) = elevation;

    metaResults.AudioName = audioName;
    metaResults.TrackNames = trackNames;
    metaResults.HRTFId = HRTF.Id;
    metaResults.RandTrackAngles = [azimuthLocation' zeros(size(azimuthLocation,2),1)];
    metaResults.Elevation = elevation;
    metaResults.SceneWidth = width;
    metaResults.AzimuthEnsembleOffset = azimuthOffset;
end