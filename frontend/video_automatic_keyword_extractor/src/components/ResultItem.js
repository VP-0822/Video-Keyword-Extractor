import React from "react";
import VideoSegmentInfo from "./VideoSegmentInfo";
const ResultItem = results => {
  let videoSegments;
  let noResult;
  if (results.results.length > 0) {
    videoSegments = results.results.map((result, index) => {
      let url = result.href;
      let caption = result.caption;
      let subtitle = result.subtitle;
      let captionKeywords = result.caption_keywords.no_video_id.multiModal.map(keyword => {
          return {
              keyword: keyword[0],
              score: keyword[1]
          }
      });
      let captionSubtitleKeywords = result.caption_subtitle_keywords.no_video_id.multiModal.map(keyword => {
        return {
            keyword: keyword[0],
            score: keyword[1]
        }
      });
      let segmentTimestamp;
      result.timestamp.forEach((time, index) => {
        if(index == 0) {
          segmentTimestamp = time + ' secs to ';
        } else {
          segmentTimestamp = segmentTimestamp + time + ' secs';
        }
      });
      
      return <div class="grid-item"><VideoSegmentInfo key={index} url={url} caption={caption} subtitle={subtitle} captionKeywords={captionKeywords} captionSubtitleKeywords={captionSubtitleKeywords} segmentInfo={segmentTimestamp} /></div>;
    });
  } else {
      noResult = <h1> Sorry, no results found. Please try again.</h1>;
  }
  return (
    <div>
      <div class="grid-container">
      <ul>{videoSegments}</ul>
      </div>
      {noResult}
    </div>
  );
};

export default ResultItem;