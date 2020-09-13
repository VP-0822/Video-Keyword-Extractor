import React from "react";
import ResultContainer from "./ResultContainer";

const Results = ({ videoId, results }) => {
  return (
    <div>
      <h2>Keywords for video id: {videoId}</h2>
      <ResultContainer videoId={videoId} results={results}/>
    </div>
  );
};

export default Results;