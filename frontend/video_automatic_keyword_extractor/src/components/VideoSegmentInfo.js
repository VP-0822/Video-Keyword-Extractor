import React from "react";
import ReactPlayer from "react-player";
import KeywordWithScore from "./KeywordWithScore";

const VideoSegmentInfo = ({url, caption, subtitle, captionKeywords, captionSubtitleKeywords, segmentInfo}) => {
  let keywordsForCaptions = captionKeywords.map((keywordWithScore, index) => {
    return <KeywordWithScore key={index} keyword={keywordWithScore.keyword} score={keywordWithScore.score}/>
  });

  let keywordsForCaptionsAndSubtitles = captionSubtitleKeywords.map((keywordWithScore, index) => {
    return <KeywordWithScore key={index} keyword={keywordWithScore.keyword} score={keywordWithScore.score}/>
  });
  return (
    <div class="card">
      <div class="segmentInfo">
        <span class="containerTitle">Segment timestamp: </span>
        <span><b>{segmentInfo}</b></span>
      </div>
      <ReactPlayer
        url={url}
        width="430px"
        className="react-player"
      />
      <div class="dataContainer">
        <p class="containerTitle">Caption:</p>
        <h4><b>{caption}</b></h4>
        <p class="containerTitle">Keywords for caption:</p>
        <ul>{keywordsForCaptions}</ul>
        <p class="containerTitle">Keywords for caption and subtitle:</p>
        <ul>{keywordsForCaptionsAndSubtitles}</ul>
        <p class="containerTitle">Subtitle:</p>
        <p>{subtitle}</p>
      </div>
    </div>
  );
};

export default VideoSegmentInfo;