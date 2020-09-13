import React from "react";
import InputForm from "./InputForm";

const Header = ({ handleSubmit }) => {
  return (
    <div>
      <h1>Automatic Video Keyword Extractor</h1>
      <InputForm handleSubmit={handleSubmit}/>
    </div>
  );
};

export default Header;