# Hard stops: store, licence, entitlement, attestation, sign-in and user-account entry points

Status: inventory only. **Nothing here is to be implemented by agents.** The repo owner decides policy
for every row marked "owner decision". Sources: `runtime/os/xgameruntime_provider.cpp` (COM slot tables,
header comment lines ~62-110), `runtime/os/xbox_offline.cpp` (C-ABI shim exports), `runtime/os/xgameruntime.cpp`,
and the comments on issue #663. No game bytes and no extracted or decrypted data are recorded here.

"Stub" in the provider means a trapping stub: if the title calls the slot, the guest stops with an
unimplemented-import report. "Offline HRESULT" means `0x800704CF` (`HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE)`,
`kOfflineHResult` in `runtime/os/xbox_offline.h`).

## 1. XStore (`xgameruntime.XStore`, 96-slot table)

| Slot | Function | Today | Set by |
|---|---|---|---|
| 3 | XStoreCreateContext | Implemented: local dummy context handle | #235 (O16) |
| 4 | XStoreCloseContextHandle | Implemented | #235 (O16) |
| 66 / 67 | XStoreRegisterGameLicenseChanged / Unregister | Trapping stub (the C-ABI shim in `xbox_offline.cpp` has a silent version, but the title path does not use it) | #235 |
| 68 / 69 | XStoreRegisterPackageLicenseLost / Unregister | Trapping stub | #235 |
| other (about 40 used slots, e.g. 20/21 AcquireLicenseForPackage, 22 IsLicenseValid, 65 QueryPackageIdentifier) | various | Trapping stub, "nothing about licences is invented" (O8 offline policy) | #235 / #77 |

C-ABI shim (`xbox_offline.cpp`, XGR exports): XStoreQueryGameLicense, QueryAddOnLicenses, QueryAssociatedProducts,
QueryProductForCurrentGame, Register/UnregisterGameLicenseChanged all complete with the offline HRESULT (no
licence data). Set by #77 (O8).

## 2. XStoreExt (`xgameruntime.XStoreExt`)

| Slot | Function | Today | Set by |
|---|---|---|---|
| 71 | XStoreAcquireLicenseForDurablesAsync(ctx, storeId, async) | Synchronous `E_NOTIMPL`, logs "offline mode", claims zero ownership; the title maps it to state 4 (unavailable) | #856 (O37 / #849); owner sign-off "Option A" on #663 |
| 72 | XStoreAcquireLicenseForDurablesResult | Same, clears the out handle | #856 |
| all other | | Trapping stub | |

Open question for the owner: the research comment on #663 lists alternatives (`E_GAMESTORE_NETWORK_ERROR 0x89245301`,
`0x803F9009`). Codes that assert ownership or sign-in state (`0x803F8001`, `0x803F9006`, `0x803F8008`) or return a
licence handle are entitlement claims and must not be used without an explicit decision.

## 3. XUser (`xgameruntime.XUser`, `XUserExt`)

| Slot | Function | Today | Set by |
|---|---|---|---|
| 3 / 4 | XUserDuplicateHandle / CloseHandle | Implemented | #235 |
| 5 | XUserCompare | Trapping stub | |
| 7 / 8 | XUserAddAsync / AddResult | Implemented: one local offline user ("OFFLINEPLAYER"), no account picker, no real sign-in | #36 (O5), #235 |
| 9 | XUserGetLocalId | Implemented | #235 |
| 10 | XUserFindUserByLocalId | Trapping stub | |
| 11 | XUserGetId (XUID) | Implemented (local value) | #235 |
| 12 | XUserFindUserById | Trapping stub | |
| 13 | XUserGetIsGuest | Implemented: not a guest | #487 (R23) |
| 14 | unidentified | Trapping stub | |
| 19 | XUserGetAgeGroup | Implemented: `--user-age-group`, default Unknown | #514 (R23) |
| 20 | XUserCheckPrivilege | Implemented: no privilege, deny reason Unknown | #235 / #77 |
| 21 / 22 | XUserResolvePrivilegeWithUiAsync / Result | Implemented: offline HRESULT | #235 / #77 |
| 23 | XUserGetTokenAndSignatureAsync | Implemented: completes with offline HRESULT, no token | #235 / #77 |
| 24 | XUserGetTokenAndSignatureResultSize | Implemented: offline HRESULT | #235 / #77 |
| 25 | XUserGetTokenAndSignatureResult | **Trapping stub** in the provider (the C-ABI shim version returns a null buffer) | #235 |
| 33 / 34 | XUserRegisterForChangeEvent / Unregister | Implemented (no events fire) | #235 |
| XUserExt 3 | XUserGetGamertag | Implemented (local name) | #235 |

Xbox Live (XSAPI, `XblContextCreateHandle` etc.) answers with the offline HRESULT through `xbox_offline.cpp` (#77).

## 4. XSystem, XGame, XPackage

| Slot | Function | Today | Set by |
|---|---|---|---|
| XSystem 3 | XSystemGetConsoleId | Implemented | #235 |
| XSystem 4 | XSystemGetXboxLiveSandboxId | Configured sandbox, else `E_NOTIMPL` | #235 |
| XSystem 5 | XSystemGetAppSpecificDeviceId | **Trapping stub** (device identity / attestation-adjacent; owner decision) | #235 |
| XGame 3 | XGameGetXboxTitleId | Implemented | #235 |
| XPackage 3/4 | GetCurrentProcessPackageIdentifier / IsPackagedProcess | Implemented: deterministic local id (32 zero digits), packaged | #235 |
| XPackage 5/6/7/9 | Installation monitor and progress | Implemented: always complete and launchable | #235 |
| XPackage 39 | XPackageEnumeratePackages | Content kind: none; game-package kind: `E_NOTIMPL` | #235 |
| XPackage 11/12/14/16/17/18/19 | UserLocale, FindChunkAvailability, ChangeChunkInstallOrder, InstallChunks(+Result), EstimateDownloadSize, UninstallChunks | Implemented for a fully installed package: locale en-US, chunks Ready, install completes at once, estimate 0 bytes, uninstall accepted (nothing removed) | R-xpackage |
| XPackage 34 | bool(name) "already installed" query (call site: one string, low byte of result, true skips a start-install branch; not an uninstall, public API not named with certainty) | true only for the title's own package identifier, false for null/other names; no licence or ownership claim | R-xpackage |
| XPackage 37/38/25/26 | MountWithUiAsync(+Result), GetMountPath, CloseMountHandle (slots read from the title's mount wrapper: identifier + XAsyncBlock, mount handle, 0x100-byte path buffer) | Implemented for the title's own installed package only: mounts at the game-data root `G:\`, no licence asserted; any other identifier is `HRESULT_FROM_WIN32(ERROR_NOT_FOUND)`. Not served (and not claimed): enumerating owned/other packages as entitlement | R-xpackage |
| XPersistentLocalStorage 3/5/6/7 | GetPathSize, GetSpaceInfo (used from real contents of the storage dir, 2048 MB quota), EnsureSpace(+Result) (E_ABORT when it does not fit) | Implemented; a too-small path buffer is `E_FAIL` | R-xpackage |
| XGameRuntimeIsFeatureAvailable | static survey: the title asks only ids 10 (XNetworking) and 20 (Events, telemetry) through one wrapper with two call sites; ids use the public zero-based enum | true only for 10 (GetConnectivityHint, offline). **Always false (hard stop, owner decides):** 20 Events, store 14, system 15, thread 17, user 18 (sign-in state), game streaming, game UI, invites, launcher, app capture, accessibility, any unknown id | R-xpackage |

XGame 4 XLaunchNewGame is no longer listed: relaunching the running title is a process restart, not a store,
licence or entitlement call. When the target is the title itself the guest stops with `kRelaunchSelfExitCode`
(`runtime/os/win32_compat.h`) and `tools/run_native_boot.py` / the browser boot relaunch it with the same
arguments; launching any other title returns `E_NOTIMPL`.

There is no package-licence or entitlement answer anywhere: no slot returns a licence handle or "owned".

## 5. What the boot actually hits (from #663)

- **XStoreExt slot 71**: the first hard stop, after about 14 s (18.4 s in the run where the memory-model fixes
  landed). After #856 it fails synchronously and the boot continues (next stop was an unrelated title assertion
  at 27-29 s, fixed in #892).
- **XUserAddAsync and XGameSaveInitializeProviderAsync**: reached past the Start Screen (the "OFFLINEPLAYER (A)" entry),
  both answered locally.
- **No store or licence call later in the boot.** A native menu-flow run to 240 s states "No store/licence call".
  I found no comment on #663 reporting a store/licence call at about 73 s; if one exists elsewhere, add it here.
- Token/signature and GetAppSpecificDeviceId: no comment on #663 reports the boot reaching them.
  Their stubs would trap on first call, so a stop at them would be visible as an unimplemented-import fault.

## 6. Owner decisions pending

1. Whether XUserGetTokenAndSignatureResult (slot 25) should match the offline HRESULT of slots 23/24 rather than trap.
2. What XSystemGetAppSpecificDeviceId should return, if anything.
3. Which HRESULT slot 71/72 should settle on long term (see section 2).
4. Any policy for XStore slots 20/21/22/65/68 if the boot reaches them.

## 7. Pending owner decisions: device id and XUser sign-in state

Behaviour-only summary of a static analysis of the title's call sites. No addresses, bytes or extracted data
are recorded. **Nothing below is implemented until the owner decides**; both slots still trap today.

### 7.1 XSystemGetAppSpecificDeviceId (XSystem slot 5)

What the title does:
- One call site, inside a step that builds a single text value from four parts. The fourth part is the console
  id if that equals the title's all-zero placeholder, otherwise the app-specific device id.
- Any error return (including "not implemented") or a missing interface: the step returns quietly and adds
  nothing. No retry, no abort, no fallback.
- On success it adds the id only if it equals an all-zero placeholder; any other value is read and discarded.
- No hashing, network send, save or key use was found for the id (static analysis; an indirect read cannot be
  fully excluded). Reference runtimes return fixed or per-configuration ids, which the title discards.

| Option | Consequence |
|---|---|
| A. Return a documented error, write nothing (recommended) | Boot continues, no device identity is claimed; matches the offline "nothing to report" pattern. |
| B. Fixed fake id | Same visible result; puts a made-up identity value into the runtime, identical for every player. |
| C. Per-installation derived id | Same visible result; introduces a stable machine identifier for no benefit. |
| D. All-zero placeholder | No benefit; feeds a value nobody reads. |
| E. Leave the trap | Boot stops (not viable). |

Recommendation: **A**. The only caller ignores errors by design.

### 7.2 XUser sign-in state for an offline local user (slots 14, 10, 12, 5; backlog item 17)

What the title does (all four are purely local lookups; none leads to a network call, and every online path
needs a token from slot 23, which already fails offline):
- Slot 14 (get sign-in state): polled from the user manager's update routine. Success with "signed in" keeps
  the user; an error or "signed out" makes the title treat the user as signed out and clear that user's
  pending items.
- Slots 10 / 12 (find user by local id / by XUID): return codes are ignored; an empty reference is tolerated.
- Slot 5 (compare users): "same user" on 0. Our duplicate-handle call yields a new handle object, so the
  answer must compare the underlying user, not the handle value.

| Item | Option | Consequence |
|---|---|---|
| Slot 14 | A. "Signed in" for the single local user, invalid-handle error otherwise (recommended) | Local session claim only; no tokens, no online access (slot 23 still fails offline). Matches all reference runtimes. |
| | B. "Signed out" or an error | Title keeps treating the user as signed out and clearing its items; likely breaks saves/settings (not verified dynamically). |
| | C. Leave the trap | Boot stops soon after the user is added. |
| Slots 10 / 12 | A. Handle on match with our local id / XUID, else "no such user" (recommended) | Consistent with slots 9 and 11. |
| | B. Always "no such user" | Tolerated, but the user manager cannot rebuild the record for our own user. |
| Slot 5 | A. "Same" when both handles are the same local user, by id (recommended) | Correct with duplicate handles. |
| | B. Compare by handle value | Duplicates count as different users; possible spurious "user changed" handling. |

Recommendation: slot 14 = signed in for the single local offline user, 10/12/5 as in the recommended rows;
no tokens, no online services.

Confidence: high for the call sites and error handling; medium for "the built text is never read" and the exact
meaning of the cleared list. No dynamic run backs this.
