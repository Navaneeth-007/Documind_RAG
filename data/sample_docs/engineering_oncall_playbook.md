# Engineering On-Call and Incident Response Playbook

## 1. Severity Levels and SLO Definitions
Incidents are categorized into four severity tiers based on customer impact and operational risk:

- **SEV-1 (Critical Outage)**: Core user-facing service is completely down, data loss is occurring, or critical customer transactions cannot complete. 
  - MTTA (Mean Time to Acknowledge): < 5 minutes.
  - Incident Commander (IC) assignment: Mandatory.
  - Status page update cadence: Every 15 minutes.
- **SEV-2 (Major Degradation)**: High-impact degradation affecting >10% of active users, latency p99 > 2000ms, or redundancy failure.
  - MTTA: < 15 minutes.
  - Status page update cadence: Every 30 minutes.
- **SEV-3 (Minor Issue)**: Non-critical feature impairment affecting <5% of traffic with available workarounds.
  - MTTA: < 1 hour during business hours.
- **SEV-4 (Low / Cosmetic)**: Minor bug with no functional business impact. Triaged during normal sprint planning.

## 2. On-Call Escalation Matrix
The primary on-call engineer is paged via PagerDuty. If the primary does not acknowledge the page within 5 minutes:
1. PagerDuty auto-escalates to the secondary on-call engineer.
2. If secondary does not acknowledge within 10 minutes, the engineering director and VP of Infrastructure are paged simultaneously.
3. For SEV-1 incidents, the on-call engineer must immediately spin up a dedicated War Room bridge (`meet.google.com/incident-war-room`) and notify `#eng-incidents`.

## 3. Deployment Rollback Procedures
When a regression is introduced by a recent release:
- **Kubernetes Automated Rollback**: Execute `kubectl rollout undo deployment/api-server -n production`.
- **Database Migrations**: If an irreversible migration has run, do not rollback code without a hotfix patch. Engage the Database Reliability Engineer (DBRE) on-call immediately.
- **Feature Flag Emergency Killswitch**: Toggle off the offending feature flag in LaunchDarkly within the `#eng-flags` control dashboard.
