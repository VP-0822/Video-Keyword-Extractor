import React, { Component } from 'react';
import './App.css';
import Header from './components/Header';
import Results from './components/Results';
import Loader from './components/Loader';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = { loading: false, apiResponse: "", videoId: ""};
  }

  handleSubmit(e, searchInput) {
    e.preventDefault();
    e.currentTarget.reset();
    let videoId = searchInput.substring(searchInput.indexOf('v=') + 2)
    this.setState({ loading: true, apiResponse: "", videoId: ""});
    this.callAPI('v_'+videoId);
  };
  
  callAPI(url) {
    fetch(`/keywords?videoId=${url}`)
        .then(response => response.json())
        .then(res => {
          this.setState({ loading: false, apiResponse: res, videoId: url })
        });
  }
  
  render() {
    return (
      <div className="App">
        <Header handleSubmit={this.handleSubmit.bind(this)}></Header>
        {this.state.loading ? <Loader/> : null}
        {this.state.videoId ? <Results videoId={this.state.videoId} results={this.state.apiResponse.results}/> : null}
      </div>
    );
  }
}

export default App;
