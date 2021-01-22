function Card(title) {
  return (
    <div className="container">
      <div className="card-header">{title.toUpperCase()}</div>
      <div className="card-body" id={`container-${title}`} />
    </div>
  )
}
}

export default Card
