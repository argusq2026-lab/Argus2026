import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Release signing, read from android/keystore.properties -- which is
// gitignored along with the keystore itself, because a signing key IS the
// app's identity: anyone holding it can publish updates that installed phones
// will accept. Generate one with `keytool -genkeypair` (see android/README.md,
// "Building an installable APK").
//
// Deliberately optional: with no keystore.properties, `assembleRelease` still
// builds and produces app-release-unsigned.apk, so CI and a fresh clone are
// not broken by the absence of a secret. What must NOT happen is a release
// quietly signed with a debug key -- that is why this reads a dedicated file
// rather than falling back to the debug keystore.
val keystoreProps = Properties().apply {
    val f = rootProject.file("keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "com.argus.edge"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.argus.edge"
        // Snapdragon 8 Elite (SM8750) ships Android 15. 31 is a safe floor that
        // still covers the CameraX APIs used here.
        minSdk = 31
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.0"

        // arm64-v8a only. The Hexagon NPU exists on no other ABI, and shipping
        // a 32-bit slice would produce a build that installs and then cannot
        // load the QNN backend -- exactly the quiet degradation this project
        // refuses everywhere else.
        ndk { abiFilters += "arm64-v8a" }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        if (keystoreProps.isNotEmpty()) {
            create("release") {
                storeFile = rootProject.file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            // Minification stays off on purpose, not as an oversight. R8 has
            // two failure modes here that both degrade silently -- stripping
            // JNI entry points onnxruntime reaches by reflection, and renaming
            // classes OkHttp touches the same way -- and neither shows up at
            // build time; the minified APK compiles fine and fails on a
            // device. The bulk of the APK is onnxruntime-qnn native libraries
            // R8 cannot touch: measured on this tree, minification saves
            // 6.8 MB of 80 (80.0 -> 73.2), and nobody has run the minified
            // build on hardware. Turn it on only with a device test behind it.
            isMinifyEnabled = false
            signingConfigs.findByName("release")?.let { signingConfig = it }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    sourceSets {
        getByName("main").java.srcDirs("src/main/kotlin")
        getByName("test").java.srcDirs("src/test/kotlin")
        getByName("androidTest").java.srcDirs("src/androidTest/kotlin")
        // The cross-language fixtures live with the Python tests that generate
        // them (tests/data). Host unit tests read them via a system property;
        // instrumented tests get them packaged as androidTest assets. One file,
        // both platforms — never two drifting copies.
        getByName("androidTest").assets.srcDirs("${rootDir.parentFile}/tests/data")
    }

    testOptions {
        unitTests.all {
            it.systemProperty("argus.fixtures", "${rootDir.parentFile}/tests/data")
            // Each exercise's model artifact is a shipped asset, not a fixture:
            // the app loads it from assets at runtime, and the host test loads
            // the very same file so a test can never pass against a copy the
            // phone would not use.
            it.systemProperty("argus.assets", "${projectDir}/src/main/assets")
        }
    }

    packaging {
        jniLibs {
            // Extract native libraries to disk at install time.
            //
            // Since API 23 the platform default is to leave uncompressed .so
            // files inside the APK and map them from there, which is why this
            // has to be asked for explicitly. QNN does not work that way: the
            // execution provider takes a filesystem `backend_path` and dlopens
            // libQnnHtp.so, which in turn loads the per-Hexagon skel onto the
            // DSP. With the libraries unextracted there is no such path, the EP
            // silently fails to register, every node falls back to CPU, and the
            // only reason that surfaces at all is that CPU fallback is
            // explicitly disabled. Found by QnnSessionTest on a real S25 Ultra.
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // ONNX Runtime with the QNN execution provider compiled in. This pulls
    // com.qualcomm.qti:qnn-runtime transitively, which is where libQnnHtp.so,
    // libQnnSystem.so and the per-Hexagon skels come from -- including
    // libQnnHtpV79Skel.so, the Snapdragon 8 Elite's. Both are on Maven Central;
    // no Qualcomm account or QAIRT SDK download is involved.
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.28.0")

    // WebSocket client for the ingest protocol (docs/PROTOCOL.md).
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    testImplementation("org.json:json:20240303")
}
