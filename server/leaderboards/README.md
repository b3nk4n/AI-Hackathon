### Endpoint

`http://bsautermeister.de/emotional-hero/api/leaderboards`

### JSON Data Structure

```
{
  user: "Player 1",
  score: 1337,
  timestamp: 1498245307
}
```

## Examples

### GET all

`http://bsautermeister.de/emotional-hero/api/leaderboards`

Returns:

```
[{
   user: "Player 1",
   score: 1337,
   timestamp: 1498245307
 },
 {
   user: "Player 2",
   score: 7331,
   timestamp: 1498245533
 },
 ...,
]
```

### GET all with score >= 1000

`http://bsautermeister.de/emotional-hero/api/leaderboards?score>=1000`

### GET all ordered by score and player

`http://bsautermeister.de/emotional-hero/api/leaderboards?sort=score:-1,user:1`

- ascending: 1
- descending: -1
