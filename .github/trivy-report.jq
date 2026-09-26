def key: "\(.PkgName)|\(.InstalledVersion)|\(.VulnerabilityID)";

(reduce ($base[0].Results[]?.Vulnerabilities[]? | key) as $k ({}; .[$k] = true)) as $in_base
| ($accepted | split("\n") | map(select(. != "" and (startswith("#") | not)))) as $accepted_ids
| [
    .Results[]? | .Target as $target | .Vulnerabilities[]?
    | select(.Severity == "HIGH" or .Severity == "CRITICAL")
    | [
        (if $in_base[key] then "base"
         elif (.VulnerabilityID | IN($accepted_ids[])) then "accepted"
         else "ours" end),
        .Severity,
        (if (.FixedVersion // "") == "" then "no" else "yes" end),
        .VulnerabilityID, .PkgName, .InstalledVersion, (.FixedVersion // ""), $target
      ]
  ]
| unique
| "| source | severity | fixable | id | package | installed | fixed | target |",
  "|---|---|---|---|---|---|---|---|",
  (.[] | "| " + join(" | ") + " |")
