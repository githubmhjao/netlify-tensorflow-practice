import React from "react"

function Card(props) {
  console.log(props)
  return (
    <div className="container">
      <div className="card-header">{props.title.toUpperCase()}</div>
      <div className="card-body" id={`container-${props.title}`} />
    </div>
  )
}

export default Card
