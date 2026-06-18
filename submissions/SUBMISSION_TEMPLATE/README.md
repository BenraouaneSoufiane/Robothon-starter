# Submission Template

Copy this folder to `submissions/<your-project-name>/` and replace the placeholders.

## Required files

| File | Purpose |
|------|---------|
| `registration.json` | Your platform-issued registration UUID and project metadata |
| `README.md` | Project summary, run instructions, and demo notes |
| Your code / assets | MuJoCo scenes, scripts, configs, etc. |
| Demo video | `demo.mp4` or a link in README |

## Registration UUID

1. Register on the official Hackathon platform.
2. Copy the UUID you receive.
3. Paste it into `registration.json`:

```json
{
  "uuid": "00000000-0000-0000-0000-000000000000",
  "participant_name": "Your Name or Team Name",
  "project_name": "Your Project Name"
}
```

4. Paste the same UUID at the top of your Pull Request description.

Submissions missing a valid UUID in both places may be rejected.

## Suggested README sections

- Project name
- Robot platform
- Task goal
- Technical approach
- Core features
- Highlights
- Current limitations
- Future improvements
- How to run
- Demo video
