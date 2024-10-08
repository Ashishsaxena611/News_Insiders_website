import React, { Component } from 'react'
import NewsItem from './NewsItem'
import PropTypes from 'prop-types'


export class News extends Component {
  static defaultProps={
    country:'us',
    pageSize: 6,
    category:'general',
  }
  static propTypes={
    country : PropTypes.string,
    pageSize : PropTypes.number,
    category: PropTypes.string,
  }
  constructor(){
    super();
    this.state={
      articles: [],
      loading: false,
      page:1

    }
  }

  async componentDidMount(){
    let url = `https://newsapi.org/v2/top-headlines?country=${this.props.country}&category=${this.props.category}&apiKey=4d54939b449340208d1d27b23dddf762&page=1&pageSize=${this.props.pageSize}`;
    let data = await fetch(url);
    let parsedData = await data.json()
    this.setState(({articles: parsedData.articles, totalResults:parsedData.totalResults}))
  }

  handlePrevClick = async () =>{
    let url=`https://newsapi.org/v2/top-headlines?country=${this.props.country}&category=${this.props.category}&apiKey=4d54939b449340208d1d27b23dddf762&page=${this.state.page-1}&pageSize=6`;
    let data= await fetch(url);
    let parsedData = await data.json()
   
    this.setState({
      page: this.state.page - 1,
      articles: parsedData.articles
    })
  }

  handleNextClick = async () =>{
    if(Math.ceil(this.state.page + 1>this.state.totalResults/6)){

    }
    else{

    
    let url=`https://newsapi.org/v2/top-headlines?country=${this.props.country}&category=${this.props.category}&apiKey=4d54939b449340208d1d27b23dddf762&page=${this.state.page + 1}&pageSize=6`;
    let data = await fetch(url);
    let parsedData = await data.json()
    

    this.setState({
      page: this.state.page + 1,
      articles: parsedData.articles
    })
  }
  }
  render() {
    return (
      <div className="container my-3">
       
        <h2 className="head">NewsMonkey - Top Headlines</h2>
        
        <div className="row">
        {this.state.articles.map((element)=>{
        return <div className="col-md-4 my-5" key={element.url}>
          
        <NewsItem title={element.title?element.title.slice(0,45):""} description={element.description?element.description.slice(0,88):""} imageUrl={element.urlToImage} newsUrl={element.url} author={element.author} date={element.publishedAt} source={element.source.name}/>
          
          </div>
        })}
          
        </div>
        <div className="d-flex justify-content-between">
        <button disabled={this.state.page<=1} type="button" className="btn btn-success" onClick={this.handlePrevClick} > &#8592; Previous</button>
        <button type="button" className="btn btn-success" onClick={this.handleNextClick}>Next &#8594;</button>
        </div>
      </div>
    )
  }
}

export default News
