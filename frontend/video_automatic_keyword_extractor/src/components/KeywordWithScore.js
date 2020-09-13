import React from "react";
const KeywordWithScore = ({keyword, score}) => {
  
  return (
    <div class="keywordTile">
      <span class="keywordSpan">{keyword}</span>
      <span class="scoreSpan">{score}</span>
    </div>
  );
};

export default KeywordWithScore;