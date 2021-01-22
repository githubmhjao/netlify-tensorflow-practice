import React from "react"

import Card from "./Card"

function App() {
  const cards = ["parameter", "scatter", "model", "train"]
  return (
    <>
      cards.map((card, i) => <Card key={i} title={card} />)
    </>
  )
}

export default App
