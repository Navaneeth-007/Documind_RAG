# IT Security, Compliance, and Data Protection Policy

## 1. Access Control and Multi-Factor Authentication (MFA)
All employee accounts accessing internal systems, cloud infrastructure (AWS, GCP, GitHub), and SaaS applications (Slack, Google Workspace) must enforce Multi-Factor Authentication (MFA). 
- Hardware security keys (e.g., YubiKey) or FIDO2/WebAuthn authenticators are mandatory for all engineering and IT personnel with production access.
- SMS-based authentication is strictly prohibited due to SIM-swapping vulnerabilities.
- Passwords must be at least 16 characters in length, randomly generated, and managed using the approved corporate password manager (1Password).

## 2. Workstation and Device Security (BYOD)
Personal devices (BYOD) may only be used to access corporate email and calendar with mobile device management (MDM) installed. 
- All company-provided laptops must have full-disk encryption (FileVault for macOS, BitLocker for Windows) enabled at all times.
- Automatic screen lock must engage after 5 minutes of inactivity.
- Operating system security updates must be applied within 7 calendar days of release.
- Storing unencrypted customer PII or production database dumps on local workstations is a direct policy violation and grounds for termination.

## 3. Data Classification and Handling
Company data is classified into four sensitivity tiers:
1. **Public**: Marketing materials, open-source code, public press releases.
2. **Internal**: General internal wikis, standard team communications.
3. **Confidential**: Financial forecasts, vendor contracts, business strategy roadmaps.
4. **Restricted (P0)**: Customer PII, payment details (PCI), encryption keys, production API tokens.

Restricted data must be encrypted both in transit (TLS 1.3) and at rest (AES-256). Production data must never be copied to staging or development environments without automated tokenization/anonymization.

## 4. Incident Response and Reporting
If an employee suspects a data breach, lost corporate laptop, or credential leak (e.g., committed API key):
- The employee must immediately report the incident to the `#security-incidents` Slack channel and email `security@company.com`.
- The incident must be reported within 1 hour of discovery.
- The Security Incident Response Team (SIRT) will initiate containment within 15 minutes for Critical (P0) incidents.
