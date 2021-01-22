import Card from "./Card"

function App() {
  const cards = ["parameter", "scatter", "model", "train"]
  return (
    <>
      cards.map(card => <Card title={card} />)
    </>
  )
}

export default App
