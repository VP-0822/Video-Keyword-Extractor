import React from "react";
import ResultItem from "./ResultItem";

const ResultContainer = ({ searchTerm, results }) => {
  return (
    <div className="photo-container">
      <ResultItem results={results}></ResultItem>
    </div>
  );
};

export default ResultContainer;