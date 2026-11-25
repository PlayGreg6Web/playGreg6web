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
| XGame 4 | XLaunchNewGame | **Trapping stub** (owner decision) | #235 |
| XPackage 3/4 | GetCurrentProcessPackageIdentifier / IsPackagedProcess | Implemented: deterministic local id (32 zero digits), packaged | #235 |
| XPackage 5/6/7/9 | Installation monitor and progress | Implemented: always complete and launchable | #235 |
| XPackage 39 | XPackageEnumeratePackages | Content kind: none; game-package kind: `E_NOTIMPL` | #235 |

There is no package-licence or entitlement answer anywhere: no slot returns a licence handle or "owned".

## 5. What the boot actually hits (from #663)

- **XStoreExt slot 71**: the first hard stop, after about 14 s (18.4 s in the run where the memory-model fixes
  landed). After #856 it fails synchronously and the boot continues (next stop was an unrelated title assertion
  at 27-29 s, fixed in #892).
- **XUserAddAsync and XGameSaveInitializeProviderAsync**: reached past the Start Screen (the "OFFLINEPLAYER (A)" entry),
  both answered locally.
- **No store or licence call later in the boot.** A native menu-flow run to 240 s states "No store/licence call".
  I found no comment on #663 reporting a store/licence call at about 73 s; if one exists elsewhere, add it here.
- Token/signature, GetAppSpecificDeviceId and XLaunchNewGame: no comment on #663 reports the boot reaching them.
  Their stubs would trap on first call, so a stop at them would be visible as an unimplemented-import fault.

## 6. Owner decisions pending

1. Whether XUserGetTokenAndSignatureResult (slot 25) should match the offline HRESULT of slots 23/24 rather than trap.
2. What XSystemGetAppSpecificDeviceId should return, if anything.
3. Whether XLaunchNewGame stays a trap.
4. Which HRESULT slot 71/72 should settle on long term (see section 2).
5. Any policy for XStore slots 20/21/22/65/68 if the boot reaches them.
