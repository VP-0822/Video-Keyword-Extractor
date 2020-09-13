import React, { useState } from "react";

const InputForm = ({ handleSubmit}) => {
  const [videoEntry, setVideoEntry] = useState("");
  // update search text state
  const updateSearchInput = e => {
    setVideoEntry(e.target.value);
  };
  return (
    <div>
      <form
        className="search-form"
        onSubmit={e => handleSubmit(e, videoEntry)}
      >
        <input
          type="text"
          name="search"
          placeholder="Enter video url..."
          onChange={updateSearchInput}
          value={videoEntry}
        />
        <button
          type="submit"
          className={`search-button ${videoEntry.trim() ? "active" : "inactive"}`}
          disabled={!videoEntry.trim()}
        >
          <svg height="32" width="32">
            <path
              d="M19.427 21.427a8.5 8.5 0 1 1 2-2l5.585 5.585c.55.55.546 1.43 0 1.976l-.024.024a1.399 1.399 0 0 1-1.976 0l-5.585-5.585zM14.5 21a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13z"
              fill="#ffffff"
              fillRule="evenodd"
            />
          </svg>
        </button>
      </form>
      <p className="examplePara"> For example: https://www.youtube.com/watch?v=bXdq2zI1Ms0</p>
    </div>
  );
};

export default InputForm;