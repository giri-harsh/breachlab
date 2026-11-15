# Lane C, L5: GitHub Actions billing lock, evidence (2026-11-13)

Scope: only what the GitHub REST API (read through the sandbox's GitHub proxy; both repos' run, job and check-run data is also
readable by anyone for the public repo) and GitHub's own documentation support. No setting was changed.

## 1. Evidence on the public repo `giri-harsh/breachlab`

Account: `giri-harsh`, type `User` (a personal account, not an organization). Repo: public, not archived, not disabled, default branch `main`.
Nine workflow runs exist in total (the repo is new); none has ever succeeded.

| Run | Workflow | Event | Result | Created (UTC) | Jobs |
|---|---|---|---|---|---|
| [43464260841](https://github.com/giri-harsh/breachlab/actions/runs/43464260841) | `probe` (one `echo` step, `runs-on: ubuntu-latest`) | workflow_dispatch | failure | 2026-11-13 16:35:35 | job `132019034029`: started 16:35:36, completed 16:35:38, `runner_id` 0, `runner_name` empty, `steps` empty |
| [43461757223](https://github.com/giri-harsh/breachlab/actions/runs/43461757223) | `canary` | schedule | failure | 2026-11-13 16:20:34 | job `132010198555` |
| [43460203465](https://github.com/giri-harsh/breachlab/actions/runs/43460203465) | `ci` | workflow_dispatch | failure | 2026-11-13 16:11:33 | jobs `132004781755` (docker), `132004782482` (web), `132004782532` (checks); `cloud-smoke` skipped |

All other runs on that repo are `skipped` (they are gated by `vars.ACTIONS_BILLING_OK`, ADR-0104).

Check-run annotations (`GET /repos/giri-harsh/breachlab/check-runs/<job_id>/annotations`) for jobs `132019034029`, `132004781755`, `132004782482`,
`132004782532` and `132010198555` are identical:

1. level `failure`, path `.github`: **"The job was not started because your account is locked due to a billing issue."**
2. level `notice`: "The ubuntu-latest label will migrate to Ubuntu 26 beginning October 19, 2026." (unrelated runner-image notice)

Reading: the job is rejected by GitHub before a runner is assigned (no runner, no steps, two seconds). The workflow content is irrelevant:
the `probe` workflow only prints `uname -a`. The lock is therefore on the account, not on the repository, the workflows or any secret.

`GET /repos/giri-harsh/breachlab/actions/permissions` returned 403 from the proxy ("Access to this GitHub Actions path is not permitted through this proxy"),
and `/users/giri-harsh` is not available from the sandbox, so the plan, spending limit and billing state of the account could not be read here; they are only visible to Harsh when signed in.

## 2. When the lock started (private archive repo `giri-harsh/Breach-lab`)

The archive repo has 404 runs. Counts by result: 236 `ci` success, 18 `canary` success, 3 `model-gate` success, 1 `drift` success (258 successful runs in total) from 2026-10-29,
and 112 `startup_failure` runs.

- Last successful run: [41920210060](https://github.com/giri-harsh/Breach-lab/actions/runs/41920210060) (`canary`, schedule) at **2026-11-04 21:53:08 UTC**.
- First `startup_failure`: [41955524418](https://github.com/giri-harsh/Breach-lab/actions/runs/41955524418) (schedule) at **2026-11-05 02:29:16 UTC**; they continue to the latest push at 2026-11-13 16:55 UTC. These runs have 0 jobs and no check-run annotation (the private repo shows "startup failure" instead of the annotated job failure).
- So CI worked on this account up to 2026-11-04 evening and broke within the following 4.6 hours. Look at Payment history and any billing e-mail around 2026-11-04 to 2026-11-05 first (a renewal, a failed card authorization, a plan change, or a Copilot/other trial charge are the causes other users report, see section 3).
- The lock therefore predates the public repo; moving to a new repo did not and cannot help.

## 3. What GitHub's documentation and public reports say

- GitHub Docs, [About billing for GitHub Actions](https://docs.github.com/en/billing/concepts/product-billing/github-actions): standard GitHub-hosted runners in public repositories are free and do not use the included amount;
  the Free plan includes a monthly amount of $12 of compute for private repositories (the page expresses it as money, not minutes); without a payment method, usage is blocked once the included amount is used.
  The page does not mention a "locked account" or this annotation text.
- GitHub Docs, [Troubleshooting workflows, "Reviewing billing errors"](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows): "Setting an Actions budget may help immediately unblock workflows failing due to billing or storage errors."
  It links to [Setting up budgets to control spending on metered products](https://docs.github.com/en/billing/how-tos/set-up-budgets).
- That budgets page: personal accounts open https://github.com/settings/billing, click **Budgets and alerts**, **New budget**; if the stop-usage option is not selected usage continues and only an e-mail is sent; its example of blocking is a budget that was used up.
  It does not say that a $0 budget or a default budget locks an account.
- No official GitHub page found describes the exact sentence "your account is locked due to a billing issue" or says how to clear it. The searches turned up only community threads
  ([#186091](https://github.com/orgs/community/discussions/186091), [#188932](https://github.com/orgs/community/discussions/188932), [#176249](https://github.com/orgs/community/discussions/176249), [#212276](https://github.com/orgs/community/discussions/212276)),
  all without a staff answer on the pages read (two were "Closed Unanswered"). Reported causes (unverified, from users): a failed card authorization or unpaid invoice, a plan downgrade with an old unpaid invoice, a failed Copilot/other trial activation.
  Reported fixes: re-add the card or retry a failed payment (did not always work), set an Actions budget (did not always work), and contacting GitHub Support, because community members cannot unlock accounts; one user wrote the lock "automatically resolved itself in a few days" after opening a ticket.
- Conclusion: nothing in code or repo settings can remove the lock. Harsh can try the billing checks below (cheap, 5 minutes); if the page shows nothing wrong, GitHub Support is the route.

## 4. Message for GitHub Support (ready to paste, also in docs/blockers.md)

> Subject: Actions jobs are not started: "your account is locked due to a billing issue" (personal account giri-harsh)
>
> Hello, every GitHub Actions job on my personal account `giri-harsh` fails within two seconds with the annotation "The job was not started because your account is locked due to a billing issue." This happens on a public repository even for a trivial workflow that only runs `echo`.
> Example run: https://github.com/giri-harsh/breachlab/actions/runs/43464260841 (job 132019034029, workflow_dispatch on 2026-11-13 16:35 UTC; runner_name is empty and there are no steps).
> The same lock affects my private repository `giri-harsh/Breach-lab`: its last successful run was https://github.com/giri-harsh/Breach-lab/actions/runs/41920210060 on 2026-11-04 21:53 UTC, and every run since 2026-11-05 02:29 UTC (for example https://github.com/giri-harsh/Breach-lab/actions/runs/41955524418) is a startup failure.
> I have checked Settings > Billing and licensing: [fill in: no failed payment / payment method valid / no budget set / the failed charge shown]. Please tell me what is outstanding on the account and unlock Actions. Thank you.
