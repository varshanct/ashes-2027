# Decisions diary

One entry per real decision: the options, what I chose and why.

---

## Reading the match data

## 2026-11-09: Use every men's Test in Cricsheet, not only England v Australia
Options: Ashes Tests only / all men's Tests.
Chose: all men's Tests.
Why: there are only a few dozen recent Ashes Tests. Results against other teams also tell the model how strong
each side is, and a Bayesian model can pool that information sensibly.

## 2026-11-09: Keep only men's Tests when reading the files
Why: the download can include other formats and women's Tests. The code skips anything that is not a men's Test
and reports how many files it skipped, so nothing is dropped silently.

## 2026-11-09: Retired hurt is not a dismissal
Why: a batter who retires hurt can come back, and the batting side has not lost a wicket. Counting it as a wicket
would make teams look weaker than they were. The same applies to "retired not out".

## 2026-11-09: Add penalty runs to the innings total
Why: penalty runs count on the scorecard but belong to no ball, so they are read separately from the innings
and added to its total. Without them, some totals would be too low.

## 2026-11-09: Work out the follow-on from the batting order
Why: Cricsheet does not flag the follow-on. If the same team bats in the 2nd and 3rd innings, it must have been
enforced, so the code infers it from that.

## 2026-11-09: Count only legal balls towards an over
Why: wides and no-balls do not count towards the six balls of an over. Ball numbers and overs bowled skip them,
which matters for measuring how much play was lost to rain.

## 2026-11-09: Save the processed tables as CSV
Options: CSV / Parquet.
Chose: CSV.
Why: I can open and check the tables in VS Code or Excel, and no extra library is needed.
I will switch if loading the ball-by-ball table becomes slow.

## 2026-11-09: Test on hand-built fake matches before real ones
Why: real matches are too big to check by hand. Small fake matches, each built around one tricky case
(innings win, follow-on, penalty runs, retired hurt, forfeit, draw, washout), have totals I worked out by hand,
so each test states exactly what the right answer is.

## 2026-11-10: An independent check on real innings totals
First attempt: check each fourth-innings target against the earlier innings totals. When I ran it on the real
data it reported 0 mismatches, but it had checked 0 chases, because Cricsheet does not record a target in Test
files. A check that cannot fail proves nothing, so I replaced it.
Replacement: the result margin, which Cricsheet records separately from the balls. For a win by runs, the
winner's total minus the loser's total must equal the margin; for a win by wickets, 10 minus the wickets lost in
the winner's final innings must equal it. On the real data it checked 729 of the 730 wins and found 0 mismatches.
The one win without a margin is the 2006 Oval Test, awarded to England, which has no margin to check.

---

## Entering my own knowledge as checked data

## 2026-11-10: Read player IDs as text, not numbers
Why: some Cricsheet IDs are all digits and start with 0. Read as numbers, "00123456" becomes 123456 and no longer
matches the player. A test gives every player an ID starting "00" and checks none change.
My first version of this test changed only one ID and still passed with the bug in place, so I rewrote it until it
failed without the fix.

## 2026-11-10: Test the checker on made-up files, not my real ones
Why: my real files were empty at first and will keep changing. A complete, correct set of made-up files lets
each test break exactly one thing and check the checker notices.

## 2026-11-10: What counts as a balanced Test XI
Rules: batting positions 1 to 11 used once each, a wicketkeeper, and at least four bowling options (bowlers or
allrounders).
Why: these catch the most likely mistakes when entering an XI by hand, without imposing a view of the ideal side.
Part-time spinners such as Root are not counted as bowling options.

## 2026-11-10: Make the checker double as a to-do list
Why: empty values produce messages such as "fill in p_rain_day for Lord's in month 6", so running the checker
always shows exactly what is left to do.

## 2026-11-10: Use the ECB's official ground names
Why: the schedule comes from the ECB announcement, which uses the sponsored names Utilita Bowl and Kia Oval.
The same names are used in every file, and linked to the historical names in a separate file.

## 2026-11-10: Climate averages, and why the model does not use them directly
No forecast is reliable months ahead, so I collected climate averages: rain days per month at each ground. But the
match data has no record of the rain during each past Test, so the model cannot learn how much rain causes how many
draws. Instead, each ground's effect on the draw band already includes its typical weather in the months it hosts
Tests. The climate table explains those effects, and can adjust the forecast in the week of each Test.

## 2026-11-10: Heathrow for the London grounds
Why: it is the Met Office's standard reference station for London, with full 1991-2020 averages. It is about 20 km
west of Lord's and The Oval and slightly warmer than central London, but its count of rain days is a sound guide.

---

## Squads, availability and cover

## 2026-11-10: Squad selection is my own judgement
I picked both XIs and reserves from recent form, team balance and my knowledge of the players, so the squad file
has no source links. The reasons for the key choices are below.

## 2026-11-10: Three separate questions for every player
Who would be picked if everyone were fit (squads file), can he play in this Test (availability file), and who
comes in if not (cover plan).
Why: availability is a fact about fitness, sanctions and age; selection is a judgement. Keeping them apart means
I can test selection choices, such as Boland instead of Lyon, as separate scenarios later.

## 2026-11-10: England's XI has no specialist spinner
Choice: four seamers, with Bethell, Root and Lawrence as spin options, and Patterson-White in reserve.
Why: the series is in English conditions in June to August, where seam bowling usually does most of the work.
Bethell gives a fifth bowling option. Patterson-White is the specialist spin cover the selectors chose for South
Africa, so he is first in line if a spinner is needed.

## 2026-11-10: Australia's XI without Marnus Labuschagne
Why: George Bailey confirmed on 9 November 2026 that Labuschagne will not be selected for the New Zealand Tests.
Inglis made a century at No. 5 in Cape Town, and Connolly is the incumbent at No. 7, so both keep their places.

## 2026-11-10: Konstas as Australia's reserve batter
Why: from Cricinfo's "The contenders: who replaces Marnus Labuschagne in the Test squad?" (10 November 2026).
He has the strongest form of the contenders (71 and 151 not out against Western Australia), the selectors are
watching the team's age profile and he is 21, and he can open or bat at No. 3. McSweeney is the next most likely;
Patterson is held back by age, and Clayton has never played a first-class match overseas.

## 2026-11-10: Murphy as Australia's reserve spinner
Why: Lyon will be 39 by the series, and the only other spin cover is part-time. Murphy was called up to replace
Lyon in the 2025-26 Ashes, so he is the established back-up.

## 2026-11-10: How the availability numbers were set
Rules: start near 0.95; lower for recent or repeated injury, long lay-offs, a pending disciplinary case or age
above 35; lower further for fast bowlers in the 3rd and 5th Tests, which each start four days after the previous
Test ends.
Each number has a source link. The riskiest are Hazlewood (missed the whole 2025-26 Ashes, then a knee injury) and
Archer (side strain and a history of stress fractures).

## 2026-11-10: Carse's pending disciplinary verdict
Why 0.75: he was charged by the Cricket Regulator after an incident in Derby in August 2026 and the verdict is
pending, while England's selector has said they plan to bring him back through the Lions. 0.75 reflects an
unknown outcome. If he is cleared I will delete the row; if he is suspended I will set the Tests he misses to 0.

## 2026-11-10: The order of the cover plan
England pace: Cook, then Baker, then Carse. Cook and Baker are in the Test squad named on 9 November; Carse is not
while his verdict is pending, and if he is unavailable the model simply skips him.
Australia pace: Boland, then Neser, then Webster. Boland is the established reserve.
Keepers: if Smith or Carey is out, Cox or Inglis takes the gloves and the first reserve batter comes in.

## 2026-11-10: Fill IDs by exact name, and refuse rather than guess
Why: copying 27 IDs by hand invites mistakes, so a helper fills each PUT_ID from the squad file. It only fills
when exactly one squad player has that exact name; otherwise it reports the problem and leaves the placeholder,
which the checker then flags. A wrong ID would silently attach the wrong player's record.

## 2026-11-10: My forecast, written before building any model
I reviewed each Test before seeing any model output.

| Test | Ground | England | Draw | Australia |
| --- | --- | --- | --- | --- |
| 1 | Trent Bridge | 0.40 | 0.17 | 0.43 |
| 2 | Lord's | 0.40 | 0.15 | 0.45 |
| 3 | Edgbaston | 0.42 | 0.18 | 0.40 |
| 4 | Utilita Bowl | 0.42 | 0.15 | 0.43 |
| 5 | Kia Oval | 0.47 | 0.13 | 0.40 |

Most likely series result: 2-2 with one draw (15%), followed by 3-2 either way (13% each).
Chance England win the series: 42%.
Chance Australia retain the urn (win or draw the series): 58%, including a drawn series at 16%.
These series figures come from combining the five Test forecasts, treating each Test as independent.

Why: Australia hold the urn and won the 2025-26 series 4-1, and they have won at Lord's in both 2015 and 2023.
But their attack will be older by June 2027 (Starc 37, Hazlewood 36, Lyon 39), Cummins has a recent back injury,
and the 3rd and 5th Tests each start four days after the previous one ends. England have won the final Test at
The Oval in each of the last two home Ashes. Rain makes draws likelier at Trent Bridge in June and Edgbaston in July.

What I will test later: whether the model beats this forecast, and whether treating the Tests as independent is
reasonable when the same players and fatigue carry from one Test to the next.

---

## How each ground plays

## 2026-11-10: Link ECB ground names to Cricsheet's names in a separate file
Why: the schedule uses the ECB's sponsored names (Utilita Bowl, Kia Oval), while Cricsheet uses the names in use
when each match was played (The Rose Bowl, Southampton; The Oval). A small link file, venue_aliases.csv, maps one
to the other. Code checks it: every ground needs at least one name, every name must exist in the match data, and
nothing may be listed twice. I chose each name by its city, which ruled out the many other Ovals worldwide.

## 2026-11-10: Neutral Tests count towards a ground's profile
Why: a Test between two other countries still shows how the pitch and conditions play: draw rate, first-innings
scores, how batters get out. Only the team records filter by team.

## 2026-11-10: Charts go in figures/, not outputs/
Why: the charts and the profile tables are results that belong in the repository and the README, so they are
saved to figures/, which Git tracks. outputs/ is ignored by Git and holds scratch files only.

## 2026-11-10: Chart choices
Draw rate and first-innings score sit in separate panels, each with its own axis, because a percentage and a run
total cannot share one scale. Wins, draws and losses use two clearly opposite colours, checked to stay
distinguishable for colour-blind readers, with a neutral grey for the draw. Every segment carries its number, so
nothing relies on colour alone, and every ground shows how many Tests its figures are based on.

## 2026-11-10: All Tests and Tests since 2019, side by side
Why: Test cricket in England has changed in recent years, with faster scoring and fewer draws. 2019, the start of
the World Test Championship, is a clear dividing line. Showing both shows whether a ground's long-run figures still
describe how it plays now.

## 2026-11-10: What the data cannot show: pace versus spin
My first plan was to split each ground's wickets between pace and spin. Checking the data showed Cricsheet does not
record a bowler's style, so that is not possible from these files. The profiles use scoring, draw rates, overs
bowled, the toss and how batters were dismissed instead.

## 2026-11-10: Findings from the ground profiles
Data: 111 Tests at the five grounds from 2003 to 2026, and the 38 of them played since 2019.

Draws have become much rarer: 22 of 111 Tests drawn overall (20%), but 5 of 38 since 2019 (13%). Edgbaston and
The Oval have had no draws since 2019 (0 of 7 and 0 of 8).

Rain does not explain draws on its own. Lord's has the highest draw rate of the well-sampled grounds (11 of 42) even
though London is one of the drier fixtures, while Trent Bridge, the wettest, has one of the lowest (2 of 18). The
model needs a separate effect for each ground as well as a weather effect.

The Utilita Bowl has only 7 Tests (4 since 2019). Its 43% draw rate is too noisy to trust on its own, so the model
should pool it with the other grounds rather than use it raw.

Batting first and winning the toss: the fair baseline is not 50%, because draws are possible; a side wins about half
of the non-drawn Tests by chance. The side batting first beat that baseline by about two standard errors at Trent
Bridge (12 of 18 won against 44% expected) and at Lord's (22 of 42 against 37%). But with ten comparisons, one or two
results that size are expected by chance, and no toss figure stands out. I will not build a fixed batting-first or
toss effect into the model.

Batting conditions: the average first-innings score at Lord's since 2019 is 253, the lowest of the five grounds,
and Lord's has the highest share of batters bowled. Trent Bridge averages 431 since 2019, from only 5 Tests.

England v Australia since 2001: England won all 3 at Trent Bridge and won both Oval Tests since 2019; Australia won
both Edgbaston Tests since 2019 and lead 3-2 at Lord's. The Utilita Bowl has never hosted an Ashes Test. These
counts match the actual series results from 2005 to 2023, a further check on the data.

What this means for the model: a draw probability that depends on the ground (which already reflects its typical weather), a time trend so
recent seasons count for more, and partial pooling so grounds with few Tests borrow strength from the others.

---

## The Bayesian team-strength model

## 2026-11-10: An ordered logistic model with a draw band
Options: model wins and draws as separate questions / one model with three ordered outcomes.
Chose: an ordered logistic model. The strength gap plus home advantage sits on one line; a band in the middle of
width w is a draw, and either side of it is a win for one team.
Why: a draw is what happens when neither side pulls clear, so it belongs between the two wins. One model keeps the
three chances consistent and always adding up to one.

## 2026-11-10: Team strength as a yearly random walk, centred each year
Each team starts at some level and takes a small random step every year; the model learns the typical step size.
Each year is centred so that 0 means an average Test side, because only differences between teams matter.
Why: teams rise and fall gradually, and a random walk lets recent results count for more without throwing away the
past.

## 2026-11-10: Work out the home side from the data
Cricsheet does not record who was at home. At each place (the city, or the venue if the city is blank) the home
side is the team that plays in at least 80% of the Tests there and clearly more than any other team. Otherwise the
Test counts as neutral.
Why: it needs no hand-made list of countries and correctly treats neutral Tests, such as finals at Lord's and The
Oval, as neutral.

## 2026-11-10: Leave out the ICC World XI
Why: it played one Test in 2005 and is not a national team, so it has no strength to estimate. That leaves 898 Tests
and 11 teams.

## 2026-11-10: A draw band with a base, a trend and a pooled venue effect
The log of the band's width is a base value, plus a trend per decade since 2019, plus an effect for each venue.
The venue effects share one spread, which the model also learns.
Why: my ground profiles showed draws falling sharply and differing between grounds. Pooling means a venue with
few Tests, such as the Utilita Bowl, is pulled towards the typical venue, while one with many, such as Lord's,
keeps its own value.

## 2026-11-10: Non-centred form for the random walk and venue effects
Each strength step and venue effect is written as a spread multiplied by a standard normal number, rather than drawn
directly.
Why: when a spread is small, drawing values directly creates a narrow funnel the sampler struggles to explore,
which causes divergences. The non-centred form removes the funnel. The fit had 0 divergences.

## 2026-11-10: Priors
Home advantage, draw base and draw trend have normal priors wide enough to allow large positive or negative values.
The starting spread of teams has a half-normal prior with scale 1, the yearly step a half-normal with scale 0.3
(teams change gradually), and the spread of venues a half-normal with scale 0.5.
Why: each prior rules out only absurd values, so with 898 Tests the data decide. Every interval in the results is
much narrower than its prior.

## 2026-11-10: Sampler settings and checks
4 chains, each with 1,000 warm-up and 1,000 kept draws, using NUTS with a target acceptance of 0.95 for small,
careful steps (raised from 0.9 after the back-test, see below). The fit took about 10 seconds.
Result: worst R-hat 1.002, smallest effective sample size 718, 0 divergences, so the posterior can be trusted.

## 2026-11-10: R-hat and effective sample size written from first principles
I wrote split R-hat and an effective sample size based on Geyer's initial positive sequence in NumPy, and tested them
against cases with known answers: independent draws, a stuck chain, a drifting chain and an AR(1) process with a
known theoretical effective sample size.
Why: so I understand exactly what each check measures, rather than trusting a number from a library.

## 2026-11-10: Forecasting 2027 by stepping the random walk forward
The model is fitted to 2026. For 2027, each posterior draw takes one more random step from 2026 using its own step
size, so the 2027 forecast carries a little more uncertainty than the 2026 estimate. A ground with no past Tests
would get a fresh effect drawn from the spread of venues.

## 2026-11-10: Series chances worked out within each posterior draw
For each draw, I list all 243 ways the five Tests can go and work out each one's chance, then average over draws.
Why: uncertainty about the teams' strength applies to every Test at once. If Australia are truly stronger, they are
stronger at Lord's and at Edgbaston. Treating the Tests as independent would ignore that link.

## 2026-11-10: What this model leaves out
It rates teams, not players, so the XIs, injuries, workloads between back-to-back Tests and the weather in the week
of each Test are not yet used. Home advantage is the same for every team. These are the next things to add.

## 2026-11-10: Findings from the model forecast
The fit is sound: worst R-hat 1.002, smallest effective sample size 718, 0 divergences.

Strength in 2026 (0 = average Test side): Australia 1.33 (90% interval 0.94 to 1.76), England 0.78 (0.40 to 1.16).
Australia are clearly, but not certainly, the stronger side; the intervals overlap only between 0.94 and 1.16.

Home advantage is 0.63 (0.51 to 0.75), slightly larger than the gap between the teams (about 0.55). At home, England
become slight favourites in every Test: 44% to 47%, against Australia's 40% to 43%.

Draws are becoming much rarer: the draw trend is -0.52 per decade (-0.71 to -0.34), so the draw band narrows by about
40% each decade. The model gives draw chances of 10% to 16%: highest at the Utilita Bowl (16%) and Lord's (15%), lowest
at Trent Bridge and the Kia Oval (10% each).

Partial pooling at the Utilita Bowl: its raw draw rate was 43% from 7 Tests, but the model rates its ground effect
(+0.23) about the same as Lord's (+0.24, raw rate 26% from 42 Tests). Its 90% interval is about 40% wider than Lord's
(1.16 against 0.81), showing the extra uncertainty from so few Tests.

Series: England win 47%, drawn series 13%, Australia win 40%, so Australia retain the urn 53% of the time.
Treating the Tests as independent changes this only slightly (drawn series 14%), because the uncertainty about the
strength gap is modest compared with the randomness within each Test.

Against my own forecast: the model is more positive about England (series win 47% against my 42%) and expects fewer
draws at Trent Bridge (10% against my 17%) and Edgbaston (11% against my 18%). I based those draws on rain, but
historically both grounds draw less often than average. I gave Australia the edge at Lord's (45%); the model gives
England a slight edge there too.

The strength chart is almost flat from 2001 to 2026: Australia stays near 1.3 and England near 0.8, while in reality
Australia dominated in the early 2000s, slumped around 2010-13 and won 4-1 in 2025-26. This is not a coding error.
The model only sees win, draw or loss, so a crushing win counts the same as a narrow one, and with about ten Tests per
team per year there is too little evidence to estimate each year separately. It therefore learns a small yearly
step (0.08) and smooths each team towards its long-run level, catching slow trends but missing short peaks and
slumps. The shaded bands are also wider than the real uncertainty about the Ashes, because strength is measured
against the average of all eleven teams, which includes sides with few, very uncertain Tests. The gap between England
and Australia, which is what the forecast uses, is better determined than either band.

What I make of it: the model's England edge rests on one home-advantage number shared by every team, and on yearly
strengths that are heavily smoothed. It cannot see that Australia's attack will be older by 2027 and has recent
injuries (Hazlewood, Cummins), nor that England have played without Stokes since June 2026. A near coin-flip fits the
last two Ashes in England, both drawn 2-2. I treat this as a sound starting point. The next version needs more
information from each Test, such as the margin of victory or runs and wickets from the ball-by-ball data, and
player-level information before I trust its small England edge.

---

## Testing the model on past series

## 2026-11-10: Back-testing on the 2019 and 2023 Ashes
Before trusting the 2027 forecast I tested the same model on the two most recent home Ashes. For each series the
model is fitted only on Tests played before its first day, and the home side is worked out from those Tests alone.
A test checks this cut-off: it fails if the model sees any Test from inside the series window.

## 2026-11-10: Scoring with log loss and the Brier score
Log loss punishes confident wrong forecasts heavily, which matters when forecasts feed a betting price. The Brier
score is gentler and bounded, so it shows whether one bad miss is driving the log loss. I report both, averaged
per Test.

## 2026-11-10: Two simple forecasts to beat
Equal thirds knows nothing. Past results gives the share of home wins, draws and away wins in all earlier Tests.
Beating equal thirds is easy. Beating past results shows that team strength and ground effects add information
beyond "home sides win more". Ten Tests is a small sample, so this is a first check, not a final verdict.

## 2026-11-10: Smaller sampler steps
Two of the back-test fits produced two or three divergences out of 4,000 draws, depending on the seed, although R-hat
and effective sample size were fine. I raised target_accept from 0.9 to 0.95, so NUTS takes smaller steps through the
tightly curved parts of the posterior, and gave the back-test fits 2,000 warm-up steps. Every fit then ran with no
divergences. I refitted the 2027 forecast with the new setting so both use the same model. No figure moved by more
than a point, and England still win the series 47% of the time. No back-test score changed by more than 0.002.

## 2026-11-10: Back-test results
Fitted only on earlier Tests (622 before the 2019 Ashes, 771 before 2023), the model beat both simple forecasts in
both series. Across the ten Tests its average log loss was 1.06, against 1.10 for past results and 1.10 for equal
thirds. Its Brier score was 0.65, against 0.67 for both.

The edge is real but small, and ten Tests is too few to rely on. The model made England favourites in every Test,
at 45% to 52%, yet England and Australia won four Tests each and two were drawn. It made England series favourites
both times (60% in 2019, 54% in 2023), and both series finished 2-2 with Australia keeping the urn. It gave a drawn
series about 15% each time.

Draws were partly right. Lord's had the highest draw chance in both series, and Lord's 2019 was drawn. Old Trafford
2023, drawn through rain, was given only 16%, because the model cannot see the weather in the week of a Test.

The forecasts barely move from one Test to the next, or from 2019 to 2023. That is the same weakness as the flat
team strengths: win, draw or loss alone carries too little information. Adding the margin of victory is the next
step, followed by this back-test again to see whether the edge grows.

---

## Adding the margin of victory

## 2026-11-10: One scale for every winning margin
Wins are recorded in runs or in wickets. I convert both to wickets: a wickets margin stays as it is, and a runs
margin is divided by the average runs per wicket across all Test innings (32.6). An innings win adds 10 wickets,
because the winner never needed its second innings. In the back-test the average uses only Tests before each series.

## 2026-11-10: The log of the margin
The model uses log(1 + margin in wickets). The difference between a 1-wicket and a 5-wicket win says a lot about two
teams; the difference between a 400-run and a 500-run win says very little. The log keeps close results apart and
squeezes the biggest wins together, so a few thrashings cannot dominate the fit.

## 2026-11-10: Margins share the team strengths, through a truncated normal
The result part of the model is unchanged. A second part says the size of each win follows a normal distribution
whose mean grows with the winner's edge (strength gap plus home advantage), cut off at 0. The result already says
who won, so cutting off at 0 means the margin adds only how much, without counting the win twice. The slope is
forced to be positive, so a stronger winner is expected to win by more.

## 2026-11-10: Margins as an option, tested side by side
I kept the results-only model and added margins as a second version. The back-test fits both on the same Tests and
scores both, so the comparison is fair. I chose which version produces the 2027 forecast from the back-test, not from
whether the new forecast looked better.

## 2026-11-10: Innings wins, a first attempt and a fix
My first version counted an innings win by its runs margin alone, so "an innings and 12 runs" looked like one of the
narrowest wins in the data. The margin slope came out at 0.03, almost no link between strength and the size of a
win, and the back-test barely moved (log loss 1.059 against 1.061). Innings wins are what strong sides inflict on weak
ones, so the most one-sided Tests were being recorded as close games. Adding 10 wickets for an innings win, to match
the rule for wins by wickets, raised the slope to 0.17 and gave a clear improvement in the back-test.

## 2026-11-10: More draws for the back-test
One back-test fit with margins had a worst R-hat of 1.012, just over the 1.01 target. Doubling the kept draws to 2,000
per chain brought every fit to 1.002 or below, with no divergences.

## 2026-11-10: Findings with margins
The back-test favours margins in both series. Across the ten Tests the log loss was 1.048 with margins against 1.061
for results only, 1.097 for past results and 1.099 for equal thirds; the Brier score was 0.637 against 0.647, 0.670
and 0.667. Before each series the margin version gave England about an even chance (48% in 2019, 50% in 2023), where
results only gave 60% and 54%. Both series finished 2-2.

On all Tests to 2026, the margin slope is 0.17 (0.14 to 0.21), so stronger winners clearly win by more, and the
yearly step in team strength rises from 0.08 to 0.15. The strength chart now shows England's peak around 2010-11 and
Australia's dip around 2010-13, which the results-only model smoothed away. It still smooths sharp changes:
Australia's 4-1 win in 2025-26 barely shows.

I use the margin version for the 2027 forecast. England remain slight favourites, but by less: a series win 45%
(down from 47%), Australia 42% and a drawn series 13%, so Australia retain the urn 55% of the time. Each Test is close
to even, with England between 43% and 46%. That is now within three points of my own forecast for the series.

---

## Testing on every year of Tests

## 2026-11-10: A back-test on every year, not only two series
Ten Ashes Tests is too few to tell two models apart, so I repeated the back-test for every year from 2015 to 2026.
For each year the model is fitted only on Tests before 1 January, with runs per wicket worked out from those Tests
alone, and it forecasts every Test that year in every country: 472 Tests in all. A Test involving a team with no
earlier Tests cannot be forecast, so it is skipped; this happened once, for Ireland's first Test in 2018. The home
side of each forecast Test comes from where the teams usually play, not from any result.

## 2026-11-10: Findings from the yearly back-test
Averaged over 472 Tests, the log loss was 0.884 for the results-only model, 0.890 with margins, 0.994 for past results
and 1.099 for equal thirds. The Brier scores were 0.520, 0.524, 0.597 and 0.667. The results-only model beat past
results on log loss in all 12 years.

Margins made no reliable difference: better in 4 years (2017, 2018, 2021, 2026), worse in 8, and slightly worse
overall. On the ten Ashes Tests margins had looked better (1.048 against 1.061), but that sample was too small, and
the larger back-test does not confirm it.

Sampler checks across the 24 fits: worst R-hat 1.012 in one fit, and 11 divergences in total, spread over four fits
with at most 6 in any one. Each score is an average over 20 to 50 Tests a year, so these do not change the conclusion.

## 2026-11-10: The results-only model produces the 2027 forecast
I said I would choose the version from the back-test. On 472 Tests the two versions are tied, so I use the simpler
one, which sees only results. It gives England a 47% chance of winning the series, Australia 40% and a drawn series
13%, so Australia retain the urn 53% of the time. The version with margins gave England 45% and Australia 42%. Margins
stay in the code as an option, and the strength chart from that version is kept, because it shows real changes in
form that the results-only model smooths away, even though they do not improve the forecasts.

---

## Player ratings

## 2026-11-10: Ratings from ball-by-ball data
Each player gets a batting rating (runs per dismissal) and a bowling rating (runs conceded per wicket) from every
Test he played before the date being forecast, so a back-test never sees the future. Retiring hurt is not a
dismissal, run-outs are not credited to the bowler, and byes and leg byes are not charged to him; wides and no-balls
are.

## 2026-11-10: Recent Tests count more
Each Test is weighted by its age, halving every two years. A Test two years ago counts half as much as one today,
which lets the ratings follow form: Steve Smith rates 43.8 against a career average of 55.3.

## 2026-11-10: Pulled towards the average for the player's batting position
A player with little data is treated as if he also had 10 dismissals, or 10 wickets, at an average level. My first
version used one batting average for everyone, which lifted tail-enders (Archer from 12.1 to 23.6) and held back
top-order newcomers. Batters are now pulled towards the top-order average (positions 1 to 7, 34.8 runs per dismissal)
or the lower-order average (8 to 11, 14.5), using each player's usual position from the order he came in to bat.

## 2026-11-10: First-class records for players with fewer than 10 Tests
Players with few Tests start from their first-class record instead of the role average, then their Test record is
added on top. Batting uses 85% of the first-class average, because players very rarely average more in Tests than
in first-class cricket; bowling uses the first-class average as it is, because Neser, Webster and Murphy all average
better in Tests than first-class. The figures are in data/manual/first_class.csv, taken from each player's
Wikipedia page, which quotes ESPNcricinfo with a date. Cricsheet has no county or Sheffield Shield data, so this
cannot be back-tested; it is a judgement, and the forecast will be shown with and without it.

## 2026-11-10: First-class records trusted in proportion to their size
My first version treated every first-class record as worth 10 wickets or dismissals, so Bethell's 18 part-time
wickets at 59.5 outweighed his 8 Test wickets and rated him 49.3 as a bowler. Each first-class record is now pulled
towards the role average in proportion to its size, assuming 1.5 dismissals per first-class match: Bethell is 42.6,
while Cook's 370 wickets and Neser's 440 keep almost all their weight.

## 2026-11-10: Ratings checked against current form
Robinson rates as England's best bowler (18.0), which matches his Test average below 19 after the 2026 Pakistan
series. Boland, Hazlewood, Starc and Cummins are Australia's best bowlers, and Root and Brook England's best
batters. Two things the ratings do not yet account for, both testable on the yearly back-test later: the strength
of the opposition, and the conditions (a player's record in England with the Dukes ball).

## 2026-11-11: Rating each XI
Each side's XI in every Test from 2004 is rated from its players' ratings as of 1 January that year, so a Test is
rated only from earlier years. Batting is the average batting rating of the XI. Bowling is the average bowling
rating of the five players who have bowled most, so part-timers do not count. A player with no earlier Test gets
the average for where he batted in that match, and the average bowling rating. 2004 is the first year because
earlier years have too few previous Tests to rate players from.

## 2026-11-11: Stronger XIs win more often
Across 875 Tests, sides with a much stronger batting XI won 58% and lost 20%, against 36% and 46% for sides with a
much weaker one. For bowling the spread is wider: 66% won and 19% lost, against 29% and 46%. Strong teams usually
pick strong XIs, so this does not yet show that XI strength adds anything beyond team strength. The yearly back-test
will decide that.

## 2026-11-11: XI strength in the model

- Added two numbers for the model to learn: how much a better batting XI is worth and how much a better bowling XI is worth, per 10 runs of average, on top of team strength.
- Each XI is rated from its players' records before 1 January of that year, so the back-test only uses information available at the time.
- Tests before 2004 have no XI ratings and count as two even XIs.
- Ran the yearly back-test with three versions on all 472 Tests from 2015 to 2026.
- Results only: log loss 0.883, Brier 0.519. With XI ratings: 0.885 and 0.521. With margins: 0.890 and 0.524.
- The XI version was better in 6 years, worse in 4 and level in 2. The difference is too small to matter.
- Most likely reason: team strength is learned from results, and results already reflect who played. The XI rating repeats what the model already knows.
- Decision: the main 2027 forecast stays results-only. The XI version forecasts just as well, so I use it for squad questions, such as what happens if a key bowler is unavailable.
- Sampler: 6 divergences across 36 fits and a worst R-hat of 1.027 in one fit. No other fit went above 1.011.