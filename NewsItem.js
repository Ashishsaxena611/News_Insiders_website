import React, { Component } from 'react'

export class NewsItem extends Component {
  
  render() {
   let {title,description,imageUrl,newsUrl,author,date,source}=this.props
    return (
      <div>
        <div className="card">
        <span class="position-absolute top-0  translate-middle badge rounded-pill bg-danger" style={{left:'90%', zIndex:'1'}}>
                     {source}
               </span>
                <img src={!imageUrl?"https://stastic.ui4free.com/public/images/figma-404-page-template-free-download_1630490004.jpg":imageUrl} className="card-img-top" alt="..." height="200px" width="350px"/>
                <div className="card-body">
                    <h5 className="card-title">{title}  </h5>
                    <p className="card-text">{description}</p>
                    <p className="card-text "><small className=" hii">By {!author?"unknown":author} on {new Date(date).toGMTString()}</small></p>
                    <a href={newsUrl} target="_blank" className="btn btn-primary">Read More</a>
                </div>
             </div>
      </div>
    )
  }
}

export default NewsItem
